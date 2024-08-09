use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::SeedableRng;
use rand::{rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::bounds::{Attention, Bound, Constraints, Violation, Window};
use crate::prelude::PlanBlueprint;
use crate::worker::{Capability, EventMarker, Worker, WorkerId, WorkerUtilization};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct EventId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Connection(pub EventId);

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct EventQueueEntry {
    pub event_id: EventId,
    pub start: f64,
    pub priority: i64,
    pub depth: isize,
}

impl Eq for EventQueueEntry {}

/// Explicit Ord implementation ensures BinaryHeap is a min-heap.
impl Ord for EventQueueEntry {
    // Lower priority events come first.
    fn cmp(&self, other: &Self) -> Ordering {
        match other.priority.cmp(&self.priority) {
            Ordering::Equal => {
                // Higher depth events come first.
                match self.depth.cmp(&other.depth) {
                    Ordering::Equal => {
                        // Earlier start times come first.
                        match self.start.partial_cmp(&other.start) {
                            Some(Ordering::Equal) => self.event_id.cmp(&other.event_id),
                            Some(x) => x,
                            None => Ordering::Equal,
                        }
                    }
                    x => x,
                }
            }
            x => x,
        }
    }
}

impl PartialOrd for EventQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct EventQueue {
    pub queue: BinaryHeap<EventQueueEntry>,
}

impl EventQueue {
    fn push_queue_entry(
        &mut self,
        event: &Event,
        events: &[Event],
        seen: &mut HashSet<EventId>,
        depth: isize,
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
            eprintln!("{:?} Dependency: {:?}", event.id, dependency);
            self.push_queue_entry(dependency, events, seen, depth + 1, Some(priority));
        }
        self.push(EventQueueEntry {
            event_id: event.id,
            start: event.start(),
            priority,
            depth,
        });
        seen.insert(event.id);

        for child in &event.children {
            let child_event = &events[child.0];
            self.push_queue_entry(child_event, events, seen, depth - 1, Some(priority));
        }
    }

    pub fn push(&mut self, entry: EventQueueEntry) {
        self.queue.push(entry);
    }

    pub fn pop(&mut self) -> Option<EventQueueEntry> {
        self.queue.pop()
    }

    pub fn from_events(events: &[Event]) -> Self {
        let queue = BinaryHeap::new();
        let mut this = Self { queue };
        let mut seen = HashSet::new();

        let mut queue_order = events.iter().collect::<Vec<_>>();
        queue_order.sort_by_key(|event| event.priority());
        for event in queue_order {
            this.push_queue_entry(event, events, &mut seen, 0, None);
        }
        this
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventSegment {
    pub start: f64,
    pub duration: f64,
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
    attention: Attention,
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
        attention: Attention,
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
            attention,
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

    pub fn attention(&self) -> Attention {
        self.attention
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

    pub fn dependencies(&self, events: &[Event]) -> Vec<EventId> {
        let mut deps = Vec::new();
        for dep in &self.dependencies {
            deps.push(dep.0);
            let dep_event = &events[dep.0 .0];
            deps.extend(dep_event.dependencies(events));
        }
        deps
    }

    pub fn depends_on(&self, event: EventId, events: &[Event]) -> bool {
        for dep in &self.dependencies {
            if dep.0 == event {
                return true;
            }
            let dep_event = &events[dep.0 .0];
            if dep_event.depends_on(event, events) {
                return true;
            }
        }
        false
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
        self.segments = windows
            .into_iter()
            .map(|window| EventSegment {
                start: window.start,
                duration: window.duration(),
            })
            .collect();
    }

    pub fn windows(&self) -> Vec<Window> {
        self.segments
            .iter()
            .map(|segment| segment.to_window())
            .collect()
    }

    /// Creates two child events from this event, splitting the duration.
    pub fn split_segment(&mut self, segment: isize, point: f64, duration: f64) -> bool {
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
                return false;
            }
            let new_segment = EventSegment {
                start: point + duration,
                duration: new_segment_duration,
            };
            // TODO: Check for segment overlap and join later segments if this is in the middle of
            // the segment list
            segment.duration = new_duration;
            self.segments.push(new_segment);
            true
        } else {
            // Otherise the segment just needs to be shifted forward past the point + duration.
            segment.start = point + duration;
            false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
        let events = blueprint.events;
        let queue = EventQueue::from_events(&events);

        PlanningPhase {
            events,
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
    #[cfg(test)]
    pub(crate) fn new(events: Vec<Event>, workers: Vec<Worker>, seed: [u8; 32]) -> Self {
        let queue = EventQueue::from_events(&events);
        let event_map = HashMap::from_iter(
            events
                .iter()
                .map(|event| (event.id, Uuid::from_u128(u128::from(event.id.0 as u64)))),
        );
        let worker_map = HashMap::from_iter(workers.iter().map(|worker| {
            (
                worker.id(),
                Uuid::from_u128(u128::from(worker.id().0 as u64)),
            )
        }));

        PlanningPhase {
            events,
            event_map,
            event_queue: queue,
            rand: StdRng::from_seed(seed),
            problems: Vec::new(),
            workers,
            worker_map,
        }
    }

    fn create_plan(&mut self) {
        let worker_ids: Vec<_> = self.workers.iter().map(|worker| worker.id()).collect();

        loop {
            let next_event = match self.event_queue.pop() {
                Some(queued_event) => queued_event,
                None => break,
            };
            eprintln!("\nEvent: {:?}", next_event);
            let mut work_plans = vec![];
            let mut event = (&self.events[next_event.event_id.0]).clone();

            let mut start_time = event.start();
            for dep in event.dependencies(&self.events) {
                let dep_event = &self.events[dep.0];
                let end = dep_event.total_window().end;
                start_time = start_time.max(end);
            }
            if let Some(parent) = event.parent {
                let parent_event = &self.events[parent.0];
                let parent_range = parent_event.total_window();
                start_time = start_time.max(parent_range.start);

                // The child event should not be scheduled for longer than the parent event.
                event.constraints = event
                    .constraints
                    .add_hard_bound(Bound::Upper(parent_range.end));
            }
            eprintln!("Start time: {}", start_time);
            event.constraints = event.constraints.add_hard_bound(Bound::Lower(start_time));
            event.set_start(start_time);

            for worker_id in worker_ids.iter() {
                let worker = &self.workers[worker_id.0];
                if worker.can_do(&event.requirements) {
                    // We clone the event so the worker can mutate it into a plan
                    let work_plan = worker.expected_job_duration(event.clone(), &self.events);
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
                    self.problems.push(Problem {
                        event_id: self.event_map[&next_event.event_id],
                        problem: format!(
                            "Worker is scheduled with event plan that violates constraints: {:?}",
                            vio
                        ),
                    });
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
        let mut utilization = Vec::new();
        for worker in &self.workers {
            let worker_id = self.worker_map[&worker.id()];
            for job in worker.jobs() {
                planned_events.push(PlannedEvent {
                    id: self.event_map[&job.id],
                    segments: job.segments,
                    worker_id,
                });
            }
            utilization.push(WorkerUtilization {
                worker_id,
                utilization_rate: worker.utilization_rate(),
            });
        }
        if self.problems.is_empty() {
            Ok(Plan {
                events: planned_events,
                utilization,
            })
        } else {
            Err((
                Plan {
                    events: planned_events,
                    utilization,
                },
                self.problems,
            ))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannedEvent {
    pub id: Uuid,
    pub worker_id: Uuid,
    pub segments: Vec<EventSegment>,
}

impl PlannedEvent {
    pub fn start(&self) -> f64 {
        self.segments.first().map_or(0.0, |segment| segment.start)
    }

    pub fn end(&self) -> f64 {
        self.segments
            .last()
            .map_or(0.0, |segment| segment.start + segment.duration)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    pub events: Vec<PlannedEvent>,
    pub utilization: Vec<WorkerUtilization>,
}

#[cfg(test)]
mod tests {
    use crate::bounds::{Attention, RepeatedBound};

    use super::*;

    #[test]
    fn test_queue() {
        let mut queue = EventQueue {
            queue: BinaryHeap::new(),
        };
        queue.push(EventQueueEntry {
            event_id: EventId(0),
            start: 0.0,
            priority: 0,
            depth: 0,
        });
        queue.push(EventQueueEntry {
            event_id: EventId(1),
            start: 1.0,
            priority: 1,
            depth: 0,
        });
        queue.push(EventQueueEntry {
            event_id: EventId(2),
            start: 2.0,
            priority: -1,
            depth: 0,
        });
        queue.push(EventQueueEntry {
            event_id: EventId(3),
            start: 3.0,
            priority: 1,
            depth: 1,
        });

        let entry = queue.pop().unwrap();
        assert_eq!(entry.event_id, EventId(2));
        let entry = queue.pop().unwrap();
        assert_eq!(entry.event_id, EventId(0));
        let entry = queue.pop().unwrap();
        assert_eq!(entry.event_id, EventId(3));
        let entry = queue.pop().unwrap();
        assert_eq!(entry.event_id, EventId(1));
    }

    #[test]
    fn test_event_durations() {
        let mut event = Event::new(
            EventId(0),
            0.0,
            10.0,
            0,
            Attention::NO_MULTITASKING,
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
            Attention::NO_MULTITASKING,
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
            Attention::NO_MULTITASKING,
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

    #[test]
    fn test_planning_phase() {
        let capabilites0 = HashSet::from_iter(vec![0].into_iter().map(|x| Capability(x)));
        let capabilites1 = HashSet::from_iter(vec![1].into_iter().map(|x| Capability(x)));
        let capabilites01 = HashSet::from_iter(vec![0, 1].into_iter().map(|x| Capability(x)));

        let event0 = Event::new(
            EventId(0),
            0.0,
            10.0,
            0,
            Attention::default(),
            None,
            capabilites0.clone(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let event1 = Event::new(
            EventId(1),
            0.0,
            10.0,
            2,
            Attention::default(),
            None,
            capabilites0.clone(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let event2 = Event::new(
            EventId(2),
            0.0,
            10.0,
            -1,
            Attention::default(),
            None,
            capabilites1.clone(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let worker0 = Worker::new(
            WorkerId(0),
            Constraints::default().add_repeated_block(RepeatedBound::new(
                7,
                Window {
                    start: 6.0,
                    end: 8.0,
                },
            )),
            capabilites0,
        );
        let worker1 = Worker::new(
            WorkerId(1),
            Constraints::default().add_repeated_block(RepeatedBound::new(
                7,
                Window {
                    start: 6.0,
                    end: 8.0,
                },
            )),
            capabilites01,
        );

        let phase = PlanningPhase::new(
            vec![event0, event1, event2],
            vec![worker0, worker1],
            [0; 32],
        );
        let plan = phase.plan().unwrap();

        for event in &plan.events {
            eprintln!("{:?}", event);
        }
        for worker in &plan.utilization {
            eprintln!("{:?}", worker);
        }
        assert_eq!(plan.events.len(), 3);
        for worker in plan.utilization {
            assert_eq!(worker.utilization_rate, 1.0);
        }
    }

    #[test]
    fn test_planning_dependencies() {
        let capabilites0 = HashSet::from_iter(vec![0].into_iter().map(|x| Capability(x)));
        let capabilites1 = HashSet::from_iter(vec![1].into_iter().map(|x| Capability(x)));
        let capabilites01 = HashSet::from_iter(vec![0, 1].into_iter().map(|x| Capability(x)));

        let event0 = Event::new(
            EventId(0),
            0.0,
            2.0,
            0,
            Attention::default(),
            None,
            capabilites0.clone(),
            Constraints::default(),
            Vec::new(),
            None,
        );

        let event1 = Event::new(
            EventId(1),
            0.0,
            2.0,
            2,
            Attention::default(),
            None,
            capabilites0.clone(),
            Constraints::default(),
            vec![Connection(EventId(0))],
            None,
        );

        let event2 = Event::new(
            EventId(2),
            0.0,
            2.0,
            -1,
            Attention::default(),
            None,
            capabilites1.clone(),
            Constraints::default(),
            vec![Connection(EventId(1))],
            None,
        );

        let event3 = Event::new(
            EventId(3),
            0.0,
            2.0,
            2,
            Attention::default(),
            None,
            capabilites1.clone(),
            Constraints::default(),
            Vec::new(),
            None,
        );
        let events = vec![event0, event1, event2, event3];

        let worker0 = Worker::new(
            WorkerId(0),
            Constraints::default().add_repeated_block(RepeatedBound::new(
                7,
                Window {
                    start: 6.0,
                    end: 8.0,
                },
            )),
            capabilites0,
        );
        let worker1 = Worker::new(
            WorkerId(1),
            Constraints::default().add_repeated_block(RepeatedBound::new(
                7,
                Window {
                    start: 6.0,
                    end: 8.0,
                },
            )),
            capabilites01,
        );

        let phase = PlanningPhase::new(events, vec![worker0, worker1], [0; 32]);
        let plan = phase.plan().unwrap();

        let zero = Uuid::from_u128(0);
        let one = Uuid::from_u128(1);
        let two = Uuid::from_u128(2);
        let three = Uuid::from_u128(3);

        for event in &plan.events {
            eprintln!("{:?}", event);
        }

        for event in &plan.events {
            match event.id {
                x if x == zero => {
                    assert_eq!(event.start(), 1.0);
                    assert_eq!(event.end(), 3.0);
                }
                x if x == one => {
                    assert_eq!(event.start(), 3.0);
                    assert_eq!(event.end(), 5.0);
                }
                x if x == two => {
                    assert_eq!(event.start(), 5.0);
                    assert_eq!(event.end(), 9.0);
                }
                x if x == three => {
                    assert_eq!(event.start(), 3.0);
                    assert_eq!(event.end(), 5.0);
                }
                _ => panic!("Unexpected event id"),
            }
        }
        for worker in &plan.utilization {
            eprintln!("{:?}", worker);
        }
        assert_eq!(plan.events.len(), 4);
        for worker in plan.utilization {
            match worker.worker_id {
                x if x == zero => assert_eq!(worker.utilization_rate, 0.5),
                x if x == one => assert_eq!(worker.utilization_rate, 1.0),
                _ => panic!("Unexpected worker id"),
            }
        }
    }
}
