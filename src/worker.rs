use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::bounds::{Bound, Consideration, Constraints, Violation, Window, DEFAULT_ATTENTION};
use crate::event::{split_segment, Event, EventId, Explanation};
use crate::prelude::Attention;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Capability(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(pub(crate) usize);

#[derive(Debug, Clone)]
pub struct EventMarker {
    worker_id: WorkerId,
    event: Event,
    interrupting: Option<EventId>,
    violations: Vec<Violation>,
    cost: f64,
}

// TODO: These constants should be configurable
pub const START_COST_ADJUSTMENT: f64 = 10.0;
pub const DURATION_COST_ADJUSTMENT: f64 = 10.0;
pub const EVENT_VIOLATION_COST: f64 = 100.0; // Arbitrary
pub const INTERRUPT_COST: f64 = 10.0; // Arbitrary
pub const ATTENTION_COST_ADJUSTMENT: f64 = 0.1;

lazy_static::lazy_static! {
    pub static ref COSTS: HashMap<String, f64> = {
        let mut m = HashMap::new();
        m.insert("start".to_string(), START_COST_ADJUSTMENT);
        m.insert("duration".to_string(), DURATION_COST_ADJUSTMENT);
        m.insert("violation".to_string(), EVENT_VIOLATION_COST);
        m.insert("interrupt".to_string(), INTERRUPT_COST);
        m.insert("attention".to_string(), ATTENTION_COST_ADJUSTMENT);
        m
    };
}

impl EventMarker {
    pub fn calc_cost(
        event: &Event,
        interrupting: bool,
        total_attention: u32,
    ) -> (f64, Vec<Violation>) {
        let mut cost = 0.0;
        cost += event.start() * START_COST_ADJUSTMENT;
        cost += event.adjusted_duration() * DURATION_COST_ADJUSTMENT;
        cost += if interrupting { INTERRUPT_COST } else { 0.0 };
        let violations = event.constraints_met();
        for _ in &violations {
            cost += EVENT_VIOLATION_COST;
        }
        cost += total_attention as f64 * ATTENTION_COST_ADJUSTMENT;
        (cost, violations)
    }

    pub fn id(&self) -> EventId {
        self.event.id()
    }

    pub fn explain(
        &self,
        worker_lookup: &HashMap<WorkerId, Uuid>,
        event_lookup: &HashMap<EventId, Uuid>,
    ) -> Explanation {
        Explanation {
            event_id: event_lookup.get(&self.event.id()).unwrap().clone(),
            worker_id: *worker_lookup.get(&self.worker_id).unwrap(),
            interrupting: self
                .interrupting
                .iter()
                .map(|i| *event_lookup.get(&i).unwrap())
                .collect(),
            violations: self.violations.clone(),
            cost: self.cost,
            start_time: self.event.start(),
            end_time: self.event.end(),
            duration: self.event.adjusted_duration(),
        }
    }

    pub fn cost(&self) -> f64 {
        self.cost
    }

    pub fn end(&self) -> f64 {
        self.event.end()
    }

    pub fn worker_id(&self) -> WorkerId {
        self.worker_id
    }

    pub fn windows(&self) -> Vec<Window> {
        self.event.windows()
    }

    pub fn violations(&self) -> &[Violation] {
        self.violations.as_slice()
    }

    pub fn new(
        worker_id: WorkerId,
        event: Event,
        interrupting: Option<EventId>,
        total_attention: u32,
    ) -> Self {
        let (cost, violations) = Self::calc_cost(&event, interrupting.is_some(), total_attention);
        Self {
            worker_id,
            event,
            interrupting,
            violations,
            cost,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Worker {
    id: WorkerId,
    blocked_off: Constraints,
    jobs: Vec<EventMarker>,
    capabilities: HashSet<Capability>,
}

impl Worker {
    /// Constructor
    pub fn new(id: WorkerId, blocked_off: Constraints, capabilities: HashSet<Capability>) -> Self {
        Self {
            id,
            blocked_off,
            jobs: Vec::new(),
            capabilities,
        }
    }

    /// Retireves the worker's indexed ID
    pub fn id(&self) -> WorkerId {
        self.id
    }

    /// Retireves all currently assigned jobs to the worker
    pub fn jobs(&self) -> Vec<Event> {
        self.jobs.iter().map(|job| job.event.clone()).collect()
    }

    /// A worker is available for jobs which they don't have a hard bound blocking
    /// e.g. (hiring, quitting)
    pub fn is_available_for(&self, range: Window) -> bool {
        self.blocked_off.hard_bound_fn()(range).is_empty()
    }

    /// A worker can handle requirements that is in their capabilities
    pub fn can_do(&self, event: &Event) -> bool {
        let requirements = event.requirements();
        let result = self.capabilities.is_superset(requirements);
        #[cfg(feature = "tracing")]
        tracing::debug!(
            "Worker({:?}) has capabilities {:?} and {:?} has requirements {:?}: {}",
            self.id,
            self.capabilities,
            event.id(),
            requirements,
            result
        );
        result
    }

    /// Add a job for the worker as taken from a job plan produced by expected_job_duration
    pub fn add_job(&mut self, event: EventMarker) {
        assert!(event.worker_id == self.id);
        #[cfg(feature = "tracing")]
        tracing::debug!(
            "Assigning {:?} to worker {:?} with cost {}",
            event.id(),
            event.worker_id(),
            event.cost()
        );

        if let Some(interrupted_work) = event.interrupting {
            let job = self.jobs.get_mut(interrupted_work.0).unwrap();
            job.event.interrupt(&event.event);
        }

        self.jobs.push(event);
    }

    /// Creates a draft working plan for the worker on this event.
    /// Assumptions:
    ///
    /// All jobs are actively being worked on.
    /// Events arrive in priority order.
    /// Events are scheduled in the order they arrive.
    pub fn expected_job_duration(&self, mut event: Event, events: &[Event]) -> Option<EventMarker> {
        let event_priority = event.priority();
        let mut interrupted_work = None;

        // Considerations are the windows of all the jobs that are currently being worked on.
        let mut considerations = vec![];
        for job in self.jobs.iter() {
            let job_priority = job.event.priority();

            if event_priority < 0 && job_priority >= 0 {
                if !event.depends_on(job.event.id(), events) {
                    // Negative priority events are classified as interrupts and can be scheduled at
                    // the same time as positive priority events.
                    interrupted_work = Some(event.id());
                    break;
                }
            }

            if let Some(parent_id) = event.parent_id() {
                if parent_id == job.event.id() {
                    // Children are technically the same job as their parent, so they can be
                    // scheduled at the same time.
                    continue;
                }
            }
            considerations.push(Consideration::new(
                job.event.total_window(),
                job.event.attention(),
            ));
        }

        // Move the event so it doesn't overlap with any hard bounds.
        for bound in self.blocked_off.hard_bounds() {
            match bound {
                Bound::Lower(expected) => {
                    if event.start() < expected {
                        event.set_start(expected);
                    }
                }
                Bound::Upper(expected) => {
                    if event.start() + event.adjusted_duration() > expected {
                        return None;
                    }
                }
            }
        }

        // Windows keeping track of the total worker attention used in this time period.
        let mut attention_windows: HashMap<Window, Attention> = HashMap::new();

        // Add a delay to the start time and duration to account for weekends, etc.
        for consideration in self.blocked_off.block_iter(event.start(), considerations) {
            if event.total_window().end < consideration.window.start {
                break;
            }

            for (i, window) in event.windows().into_iter().enumerate() {
                if window.overlap(consideration.window) {
                    // Note: the way this is set up PARTIAL_MULTITASKING can only multitask with
                    // PARITAL_MULTITASKING or lower and FULL_MULTITASKING will multitask with
                    // anything but NO_MULTITASKING.
                    let current_attention = *attention_windows
                        .entry(window)
                        .or_insert_with(|| event.attention());
                    if !consideration
                        .attention
                        .can_multitask_with(&current_attention)
                    {
                        attention_windows.remove(&window);
                        if split_segment(
                            &mut event.segments,
                            i as isize,
                            consideration.window.start,
                            consideration.window.duration(),
                        ) {
                            let windows = event.windows();
                            attention_windows.insert(windows[i], current_attention);
                            attention_windows.insert(windows[i + 1], current_attention);
                        } else {
                            // The window can shift to the right, so we need to update the window
                            let windows = event.windows();
                            attention_windows.insert(windows[i], current_attention);
                        }
                    } else {
                        attention_windows
                            .insert(window, current_attention + consideration.attention);
                    }
                }
            }
        }

        // Sum all attentional windows so we prefer schedules with less distracted workers with
        // fewer breaks between their attention usages.
        let mut total_attention: u32 = 0;
        for attn in attention_windows.values() {
            total_attention += attn.value() as u32;
        }

        let range = event.total_window();
        for bound in self.blocked_off.hard_bounds() {
            if bound.violated(range).is_some() {
                return None;
            }
        }
        Some(EventMarker::new(
            self.id,
            event,
            interrupted_work,
            total_attention,
        ))
    }

    /// Computes the total work time vs utilized work time as a ratio between [0.0, 1.33]
    /// 0.0 means the worker is never working, 1.0 means the worker is always working, and 1.33
    /// means the worker is overloaded.
    pub fn utilization_rate(&self) -> f64 {
        let mut jobs = self.jobs();
        // Just in case
        jobs.sort_by(|a, b| a.start().partial_cmp(&b.start()).unwrap());

        let mut start_time: f64 = 0.0;
        let mut end_time: f64 = 0.0;
        let mut working_time: f64 = 0.0;
        let mut blocked_off_time: f64 = 0.0;

        for bound in self.blocked_off.hard_bounds() {
            match bound {
                Bound::Lower(lower) => start_time = lower,
                _ => {}
            }
        }

        // Total time of all the jobs
        for job in jobs {
            let window = job.total_window();
            let attention = job.attention().value() as f64 / DEFAULT_ATTENTION as f64;
            end_time = end_time.max(window.end);
            for window in job.windows() {
                working_time += window.duration() * attention;
            }
        }

        for consideration in self.blocked_off.block_iter(start_time, vec![]) {
            if consideration.window.start >= end_time {
                break;
            }
            if consideration.window.start <= start_time && consideration.window.end >= start_time {
                start_time = consideration.window.end;
                continue;
            }
            blocked_off_time += consideration.window.duration();
        }
        let total_time = end_time - start_time - blocked_off_time;

        if total_time == 0.0 {
            return 0.0;
        }
        working_time / total_time
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerUtilization {
    pub worker_id: Uuid,
    pub utilization_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bounds::{Attention, RepeatedBound},
        event::{Event, EventId},
    };

    #[test]
    fn test_utilization_time() {
        let constraints = Constraints::new()
            .add_hard_bound(Bound::Lower(1.0))
            .add_block(Window {
                start: 6.0,
                end: 8.0,
            })
            .add_block(Window {
                start: 13.0,
                end: 15.0,
            })
            .add_block(Window {
                start: 20.0,
                end: 22.0,
            });
        let mut worker = Worker::new(WorkerId(0), constraints, HashSet::new());
        let event = Event::new(
            EventId(0),
            0.0,
            4.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        worker.add_job(worker.expected_job_duration(event, &[]).unwrap());

        let event = Event::new(
            EventId(1),
            0.0,
            4.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        worker.add_job(worker.expected_job_duration(event, &[]).unwrap());

        let event = Event::new(
            EventId(2),
            15.0,
            10.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        worker.add_job(worker.expected_job_duration(event, &[]).unwrap());

        let utilization = worker.utilization_rate();
        assert_eq!(utilization, 18.0 / 20.0);
    }

    #[test]
    fn test_worker_expected_job_duration_base() {
        let mut worker = Worker::new(WorkerId(0), Constraints::new(), HashSet::new());

        let event = Event::new(
            EventId(0),
            0.0,
            10.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let job = worker.expected_job_duration(event, &[]).unwrap();
        assert_eq!(job.event.start(), 0.0);
        assert_eq!(job.event.adjusted_duration(), 10.0);
        assert_eq!(job.cost(), 100.0);
        worker.add_job(job);

        let event = Event::new(
            EventId(1),
            20.0,
            30.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let job = worker.expected_job_duration(event, &[]).unwrap();
        assert_eq!(job.event.start(), 20.0);
        assert_eq!(job.event.adjusted_duration(), 30.0);
        assert_eq!(job.cost(), 500.0);
        worker.add_job(job);

        // Add a job that should fit in between the two.
        let event = Event::new(
            EventId(2),
            0.0,
            5.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let job = worker.expected_job_duration(event, &[]).unwrap();
        assert_eq!(job.event.start(), 10.0);
        assert_eq!(job.event.adjusted_duration(), 5.0);
        assert_eq!(job.cost(), 157.5);
    }

    #[test]
    fn test_worker_expected_job_duration_constraints() {
        let constraints = Constraints::new()
            .add_hard_bound(Bound::Lower(5.0))
            .add_hard_bound(Bound::Upper(15.0));
        let worker = Worker::new(WorkerId(0), constraints, HashSet::new());
        let event = Event::new(
            EventId(0),
            0.0,
            10.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        let job = worker.expected_job_duration(event, &[]).unwrap();
        assert_eq!(job.event.start(), 5.0);
        assert_eq!(job.event.adjusted_duration(), 10.0);
        assert_eq!(job.cost(), 150.0);
        let event = Event::new(
            EventId(0),
            0.0,
            11.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        assert!(worker.expected_job_duration(event, &[]).is_none());
    }

    #[test]
    fn test_worker_expected_job_duration_weekends() {
        let constraints = Constraints::new()
            .add_repeated_block(RepeatedBound::new(
                7,
                Window {
                    start: 6.0,
                    end: 8.0,
                },
            ))
            .add_hard_bound(Bound::Upper(32.0));

        let mut worker = Worker::new(WorkerId(0), constraints, HashSet::new());
        let event = Event::new(
            EventId(0),
            0.0,
            15.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        let job = worker.expected_job_duration(event, &[]).unwrap();
        assert_eq!(job.event.start(), 1.0);
        assert_eq!(job.event.adjusted_duration(), 19.0);
        assert_eq!(job.cost(), 222.5);
        worker.add_job(job);

        let event = Event::new(
            EventId(1),
            0.0,
            5.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        let job = worker.expected_job_duration(event, &[]).unwrap();
        assert_eq!(job.event.start(), 22.0);
        assert_eq!(job.event.adjusted_duration(), 5.0);
        worker.add_job(job);

        let event = Event::new(
            EventId(1),
            0.0,
            10.0,
            0,
            Attention::DEFAULT_MULTITASKING,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        assert!(worker.expected_job_duration(event, &[]).is_none());
    }
}
