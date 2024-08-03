use std::collections::HashSet;

use crate::bounds::{Bound, Constraints, Window};
use crate::event::Event;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Capability(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(pub(crate) usize);

#[derive(Debug, Clone)]
pub struct EventMarker {
    worker_id: WorkerId,
    event: Event,
    interrupting: Option<usize>,
    cost: f64,
}

pub const START_COST_ADJUSTMENT: f64 = 1.0;
pub const DURATION_COST_ADJUSTMENT: f64 = 1.0;
pub const EVENT_VIOLATION_COST: f64 = 100.0;
pub const INTERRUPT_COST: f64 = 10.0;

impl EventMarker {
    pub fn calc_cost(event: &Event, interrupting: bool) -> f64 {
        let mut cost = 0.0;
        cost += event.start() * START_COST_ADJUSTMENT;
        cost += event.adjusted_duration() * DURATION_COST_ADJUSTMENT;
        cost += if interrupting { INTERRUPT_COST } else { 0.0 };
        let violations = event.constraints_met();
        for _ in violations {
            cost += EVENT_VIOLATION_COST;
        }
        cost
    }

    pub fn cost(&self) -> f64 {
        self.cost
    }

    pub fn worker_id(&self) -> WorkerId {
        self.worker_id
    }

    pub fn new(worker_id: WorkerId, event: Event, interrupting: Option<usize>) -> Self {
        let cost = Self::calc_cost(&event, interrupting.is_some());
        Self {
            worker_id,
            event,
            interrupting,
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
    pub fn new(id: WorkerId, blocked_off: Constraints, capabilities: HashSet<Capability>) -> Self {
        Self {
            id,
            blocked_off,
            jobs: Vec::new(),
            capabilities,
        }
    }

    pub fn id(&self) -> WorkerId {
        WorkerId(0)
    }

    pub fn jobs(&self) -> Vec<Event> {
        self.jobs.iter().map(|job| job.event.clone()).collect()
    }

    pub fn is_available_for(&self, range: Window) -> bool {
        self.blocked_off.hard_bound_fn()(range).is_empty()
    }

    pub fn can_do(&self, requirements: &HashSet<Capability>) -> bool {
        self.capabilities.is_superset(requirements)
    }

    pub fn add_job(&mut self, event: EventMarker) {
        assert!(event.worker_id == self.id);

        if let Some(interrupted_work) = event.interrupting {
            let job = self.jobs.get_mut(interrupted_work).unwrap();
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
    pub fn expected_job_duration(&self, mut event: Event) -> Option<EventMarker> {
        let event_priority = event.priority();
        let mut interrupted_work = None;

        // Considerations are the windows of all the jobs that are currently being worked on.
        let mut considerations = vec![];
        for (i, job) in self.jobs.iter().enumerate() {
            let job_priority = job.event.priority();

            if event_priority < 0 && job_priority >= 0 {
                // Negative priority events are classified as interrupts and can be scheduled at
                // the same time as positive priority events.
                interrupted_work = Some(i);
                break;
            }

            considerations.push(job.event.total_window());
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

        // Add a delay to the start time and duration to account for weekends, etc.
        for range in self.blocked_off.block_iter(event.start(), considerations) {
            if event.total_window().end < range.start {
                break;
            }

            for (i, window) in event.windows().into_iter().enumerate() {
                if window.overlap(range) {
                    event.split_segment(i as isize, range.start, range.end - range.start);
                }
            }
        }

        let range = event.total_window();
        for bound in self.blocked_off.hard_bounds() {
            if bound.violated(range).is_some() {
                return None;
            }
        }

        Some(EventMarker::new(self.id, event, interrupted_work))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bounds::RepeatedBound,
        event::{Event, EventId},
    };

    #[test]
    fn test_worker_expected_job_duration_base() {
        let mut worker = Worker::new(WorkerId(0), Constraints::new(), HashSet::new());

        let event = Event::new(
            EventId(0),
            0.0,
            10.0,
            0,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let job = worker.expected_job_duration(event).unwrap();
        assert_eq!(job.event.start(), 0.0);
        assert_eq!(job.event.adjusted_duration(), 10.0);
        assert_eq!(job.cost(), 10.0);
        worker.add_job(job);

        let event = Event::new(
            EventId(1),
            20.0,
            30.0,
            0,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let job = worker.expected_job_duration(event).unwrap();
        assert_eq!(job.event.start(), 20.0);
        assert_eq!(job.event.adjusted_duration(), 30.0);
        assert_eq!(job.cost(), 50.0);
        worker.add_job(job);

        // Add a job that should fit in between the two.
        let event = Event::new(
            EventId(2),
            0.0,
            5.0,
            0,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let job = worker.expected_job_duration(event).unwrap();
        assert_eq!(job.event.start(), 10.0);
        assert_eq!(job.event.adjusted_duration(), 5.0);
        assert_eq!(job.cost(), 15.0);
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
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        let job = worker.expected_job_duration(event).unwrap();
        assert_eq!(job.event.start(), 5.0);
        assert_eq!(job.event.adjusted_duration(), 10.0);
        assert_eq!(job.cost(), 15.0);
        let event = Event::new(
            EventId(0),
            0.0,
            11.0,
            0,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        assert!(worker.expected_job_duration(event).is_none());
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
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        let job = worker.expected_job_duration(event).unwrap();
        assert_eq!(job.event.start(), 1.0);
        assert_eq!(job.event.adjusted_duration(), 19.0);
        assert_eq!(job.cost(), 20.0);
        worker.add_job(job);

        let event = Event::new(
            EventId(1),
            0.0,
            5.0,
            0,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        let job = worker.expected_job_duration(event).unwrap();
        assert_eq!(job.event.start(), 20.0);
        assert_eq!(job.event.adjusted_duration(), 5.0);
        worker.add_job(job);

        let event = Event::new(
            EventId(1),
            0.0,
            10.0,
            0,
            None,
            HashSet::new(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        assert!(worker.expected_job_duration(event).is_none());
    }
}
