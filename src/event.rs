use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::SeedableRng;
use rand::{rngs::StdRng, seq::SliceRandom};
use uuid::Uuid;

use crate::bounds::{Bound, Constraints, Violation, Window};
use crate::prelude::PlanBlueprint;
use crate::worker::{Capability, EventMarker, Worker, WorkerId};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct EventId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Connection(pub EventId);

#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct EventQueueEntry {
    event_id: EventId,
    start: f64,
    priority: i64,
    depth: usize,
}

impl Eq for EventQueueEntry {}

/// Explicit Ord implementation ensures BinaryHeap is a min-heap.
impl Ord for EventQueueEntry {
    // Lower priority events come first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| {
                other
                    .start
                    .partial_cmp(&self.start)
                    .unwrap_or(Ordering::Equal)
            })
    }
}

struct EventQueue {
    queue: BinaryHeap<EventQueueEntry>,
}

impl EventQueue {
    fn push_queue_entry(
        queue: &mut BinaryHeap<EventQueueEntry>,
        event: &Event,
        events: &[Event],
        seen: &mut HashSet<EventId>,
        depth: usize,
        parent_priority: Option<i64>,
    ) {
        if seen.contains(&event.id) {
            return;
        }
        let priority = if let Some(parent_priority) = parent_priority {
            event.priority.min(parent_priority)
        } else {
            event.priority
        };
        for connection in &event.dependencies {
            let dependency = &events[connection.0 .0];
            Self::push_queue_entry(queue, dependency, events, seen, depth + 1, Some(priority));
        }
        queue.push(EventQueueEntry {
            event_id: event.id,
            start: event.start(),
            priority,
            depth,
        });
        seen.insert(event.id);

        for child in &event.children {
            let child_event = &events[child.0];
            Self::push_queue_entry(queue, child_event, events, seen, depth + 1, Some(priority));
        }
    }

    pub fn pop(&mut self) -> Option<EventQueueEntry> {
        self.queue.pop()
    }

    pub fn from_events(events: &[Event]) -> Self {
        let mut queue = BinaryHeap::new();
        let mut seen = HashSet::new();

        for event in events.iter() {
            Self::push_queue_entry(&mut queue, event, events, &mut seen, 0, None);
        }
        EventQueue { queue }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventSegment {
    start: f64,
    duration: f64,
}

impl EventSegment {
    pub fn new(start: f64, duration: f64) -> Self {
        EventSegment { start, duration }
    }

    pub fn to_window(&self) -> Window {
        Window {
            start: self.start,
            end: self.start + self.duration,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    id: EventId,
    priority: i64,
    assigned_worker: Option<WorkerId>,
    requirements: HashSet<Capability>,
    constraints: Constraints,
    dependencies: Vec<Connection>,
    parent: Option<EventId>,
    children: Vec<EventId>,
    segments: Vec<EventSegment>,
}

impl Event {
    pub fn new(
        id: EventId,
        start: f64,
        duration: f64,
        priority: i64,
        assigned_worker: Option<WorkerId>,
        requirements: HashSet<Capability>,
        constraints: Constraints,
        dependencies: Vec<Connection>,
        parent: Option<EventId>,
    ) -> Self {
        let segment = EventSegment { start, duration };
        Event {
            id,
            priority,
            assigned_worker,
            requirements,
            constraints,
            dependencies,
            parent,
            children: Vec::new(),
            segments: vec![segment],
        }
    }

    pub fn id(&self) -> EventId {
        self.id
    }

    pub fn parent_id(&self) -> Option<EventId> {
        self.parent
    }

    pub fn start(&self) -> f64 {
        self.segments.first().map_or(0.0, |segment| segment.start)
    }

    pub fn set_start(&mut self, start: f64) {
        let offset = start - self.start();
        for seg in self.segments.iter_mut() {
            seg.start += offset;
        }
    }

    pub fn duration(&self) -> f64 {
        self.segments
            .iter()
            .fold(0.0, |acc, segment| acc + segment.duration)
    }

    pub fn priority(&self) -> i64 {
        self.priority
    }

    pub fn dependencies(&self) -> Vec<EventId> {
        self.dependencies.iter().map(|dep| dep.0).collect()
    }

    pub fn adjusted_duration(&self) -> f64 {
        self.segments
            .last()
            .map_or(0.0, |segment| segment.start + segment.duration)
            - self.segments.first().map_or(0.0, |segment| segment.start)
    }

    pub fn interrupt(&mut self, event: &Event) {
        self.split_segment(-1, event.start(), event.duration());
    }

    pub fn total_window(&self) -> Window {
        Window {
            start: self.start(),
            end: self.start() + self.adjusted_duration(),
        }
    }

    pub fn constraints_met(&self) -> Vec<Violation> {
        self.constraints.hard_bound_fn()(self.total_window())
    }

    pub fn from_windows(&mut self, windows: Vec<Window>) {
        self.segments = windows.into_iter()
            .map(|window| EventSegment{start: window.start, duration: window.end - window.start})
            .collect();
    }

    pub fn windows(&self) -> Vec<Window> {
        self.segments
            .iter()
            .map(|segment| segment.to_window())
            .collect()
    }

    /// Creates two child events from this event, splitting the duration.
    pub fn split_segment(&mut self, segment: isize, point: f64, duration: f64) {
        assert!(duration > 0.0);

        let segment: usize = if segment < 0 {
            self.segments.len().saturating_sub(segment.abs() as usize)
        } else {
            segment as usize
        };
        let segment = self.segments.get_mut(segment).unwrap();

        let new_duration = point - segment.start;
        let new_segment_duration = segment.duration - new_duration;

        // Point is after the start of the segment.
        if new_duration > 0.0 {
            // But nothing needs to happen if the duration is zero.
            if new_segment_duration <= 0.0 {
                return;
            }
            let new_segment = EventSegment {
                start: point + duration,
                duration: new_segment_duration,
            };
            segment.duration = new_duration;
            self.segments.push(new_segment);
        } else {
            // Otherise the segment just needs to be shifted forward past the point + duration.
            segment.start = point + duration;
        }
    }
}

pub struct Problem {
    pub event_id: Uuid,
    pub problem: String,
}

pub struct PlanningPhase {
    events: Vec<Event>,
    event_map: HashMap<EventId, Uuid>,
    event_queue: EventQueue,
    rand: StdRng,

    problems: Vec<Problem>,
    workers: Vec<Worker>,
    worker_map: HashMap<WorkerId, Uuid>,
}

impl From<PlanBlueprint> for PlanningPhase {
    fn from(blueprint: PlanBlueprint) -> Self {
        let queue = EventQueue::from_events(&blueprint.events);

        PlanningPhase {
            events: blueprint.events,
            event_map: blueprint.event_map,
            event_queue: queue,
            rand: StdRng::from_seed(blueprint.seed),
            problems: Vec::new(),
            workers: blueprint.workers,
            worker_map: blueprint.worker_map,
        }
    }
}

impl PlanningPhase {
    fn create_plan(&mut self) {
        let worker_ids: Vec<_> = self.workers.iter().map(|worker| worker.id()).collect();

        loop {
            let next_event = match self.event_queue.pop() {
                Some(queued_event) => queued_event,
                None => break,
            };

            let mut work_plans = vec![];
            let mut event = (&self.events[next_event.event_id.0]).clone();

            let mut start_time = event.start();
            for dep in event.dependencies() {
                let dep_event = &self.events[dep.0];
                let end = dep_event.total_window().end;
                start_time = start_time.max(end);
            }
            if let Some(parent) = event.parent {
                let parent_event = &self.events[parent.0];
                let parent_range = parent_event.total_window();
                start_time = start_time.max(parent_range.start);

                // The child event should not be scheduled for longer than the parent event.
                event.constraints = event.constraints.add_hard_bound(Bound::Upper(parent_range.end));
            }
            event.set_start(start_time);

            for worker_id in worker_ids.iter().cloned() {
                let worker = &self.workers[worker_id.0];
                if worker.can_do(&event.requirements) {
                    // We clone the event so the worker can mutate it into a plan
                    let work_plan = worker.expected_job_duration(event.clone());
                    if let Some(work_plan) = work_plan {
                        work_plans.push(work_plan);
                    }
                }
            }
            // Randomize the order of the work plans to prevent bias.
            work_plans.shuffle(&mut self.rand);

            let mut lowest_cost: Option<EventMarker> = None;
            for work_plan in work_plans {
                if let Some(plan) = &lowest_cost {
                    if work_plan.cost() < plan.cost() {
                        lowest_cost = Some(work_plan);
                    }
                } else {
                    lowest_cost = Some(work_plan);
                }
            }

            if let Some(lowest_cost) = lowest_cost {
                let event = self.events.get_mut(next_event.event_id.0).unwrap();
                event.assigned_worker = Some(lowest_cost.worker_id());
                event.from_windows(lowest_cost.windows());
                let worker = self.workers.get_mut(lowest_cost.worker_id().0).unwrap();
                let vio = lowest_cost.violations();
                if !vio.is_empty() {
                    self.problems.push(
                        Problem {
                            event_id: self.event_map[&next_event.event_id],
                            problem: format!("Worker is scheduled with event plan that violates constraints: {:?}", vio)
                        }
                    );
                }
                worker.add_job(lowest_cost);
            } else {
                self.problems.push(Problem {
                    event_id: self.event_map[&event.id],
                    problem: "No worker available".to_string(),
                });
            }
        }
    }

    pub fn plan(mut self) -> Result<Plan, (Plan, Vec<Problem>)> {
        self.create_plan();

        let mut planned_events = Vec::new();
        for worker in &self.workers {
            for job in worker.jobs() {
                planned_events.push(PlannedEvent {
                    id: self.event_map[&job.id],
                    segments: job.segments,
                    worker_id: self.worker_map[&worker.id()],
                });
            }
        }
        if self.problems.is_empty() {
            Ok(Plan(planned_events))
        } else {
            Err((Plan(planned_events), self.problems))
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlannedEvent {
    pub id: Uuid,
    pub worker_id: Uuid,
    pub segments: Vec<EventSegment>,
}

pub struct Plan(pub Vec<PlannedEvent>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_durations() {
        let mut event = Event::new(
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

        assert_eq!(event.start(), 0.0);
        assert_eq!(event.duration(), 10.0);

        event.split_segment(0, 5.0, 5.0);
        assert_eq!(event.start(), 0.0);
        assert_eq!(event.duration(), 10.0);
        assert_eq!(event.adjusted_duration(), 15.0);
        assert_eq!(
            event.windows(),
            vec![
                Window {
                    start: 0.0,
                    end: 5.0
                },
                Window {
                    start: 10.0,
                    end: 15.0
                }
            ]
        );
    }

    #[test]
    fn test_event_before() {
        let mut event = Event::new(
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
        event.split_segment(0, -1.0, 2.0);

        // The split happens before the start of the event.
        // The event should be shifted forward by 1.0.
        assert_eq!(event.start(), 1.0);
        assert_eq!(event.adjusted_duration(), 10.0);
        assert_eq!(
            event.windows(),
            vec![Window {
                start: 1.0,
                end: 11.0,
            },]
        );
    }

    #[test]
    fn test_event_multi_split() {
        let mut event = Event::new(
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
        event.split_segment(-1, -1.0, 2.0);
        event.split_segment(-1, 6.0, 2.0);
        event.split_segment(-1, 13.0, 2.0);

        assert_eq!(event.start(), 1.0);
        assert_eq!(
            event.windows(),
            vec![
                Window {
                    start: 1.0,
                    end: 6.0
                },
                Window {
                    start: 8.0,
                    end: 13.0
                },
                Window {
                    start: 15.0,
                    end: 20.0
                },
            ]
        );
        assert_eq!(event.adjusted_duration(), 19.0);
    }
}
