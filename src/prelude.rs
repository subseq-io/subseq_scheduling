use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result as AnyResult};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use crate::bounds::{
    Attention, BlockIterator, Bound, Consideration, Constraint, Constraints, RepeatedBound, Window,
};
pub use crate::event::{split_segment, EventSegment, Plan, PlannedEvent, Problem};
use crate::event::{Connection, Event, EventId, PlanningPhase};
pub use crate::worker::{Capability, WorkerUtilization};
use crate::worker::{Worker, WorkerId};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionBlueprint(Uuid);

/// A blueprint for events that need to be planned.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventBlueprint {
    pub id: Uuid,
    pub start: Option<f64>,
    pub duration: f64,
    /// Out of 100
    pub attention: Attention,
    /// Lower is higher priority
    pub priority: i64,
    pub assigned_worker: Option<Uuid>,
    pub requirements: HashSet<Capability>,
    pub constraints: Constraints,
    pub dependencies: Vec<ConnectionBlueprint>,
    pub children: Vec<EventBlueprint>,
}

impl EventBlueprint {
    pub fn new(id: Uuid, duration: f64, priority: i64) -> Self {
        EventBlueprint {
            id,
            start: None,
            duration,
            attention: Attention::default(),
            priority,
            assigned_worker: None,
            requirements: HashSet::new(),
            constraints: Constraints::new(),
            dependencies: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn start(mut self, start: f64) -> Self {
        self.start = Some(start);
        self
    }

    pub fn attention(mut self, attention: Attention) -> Self {
        self.attention = attention;
        self
    }

    pub fn assigned_worker(mut self, assigned_worker: Uuid) -> Self {
        self.assigned_worker = Some(assigned_worker);
        self
    }

    pub fn requirements(mut self, requirements: HashSet<Capability>) -> Self {
        self.requirements = requirements;
        self
    }

    pub fn constraints(mut self, constraints: Constraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn dependency(mut self, event_id: Uuid) -> Self {
        self.dependencies.push(ConnectionBlueprint(event_id));
        self
    }

    pub fn child(mut self, mut child: EventBlueprint) -> AnyResult<Self> {
        if self.duration < child.duration {
            return Err(anyhow!("Child duration is longer than parent"));
        }
        if let Some(start) = self.start {
            if let Some(child_start) = child.start {
                if child_start < start {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("Child {:?} starts before parent {:?}", child.id, self.id);
                    child.start = Some(start);
                }
            }
        }
        // If the child has a higher priority, the parent should have the same priority
        if self.priority > child.priority {
            self.priority = child.priority;
        }
        self.children.push(child);
        Ok(self)
    }
}

pub struct PlanBlueprint {
    pub(crate) events: Vec<Event>,
    pub(crate) event_map: HashMap<EventId, Uuid>,
    pub(crate) events_seen: HashMap<Uuid, EventId>,

    pub(crate) stop_at: Option<f64>,
    pub(crate) stop_exclude: Vec<Uuid>,

    pub(crate) seed: [u8; 32],
    pub(crate) workers: Vec<Worker>,
    pub(crate) worker_map: HashMap<WorkerId, Uuid>,
}

impl PlanBlueprint {
    pub fn new(seed: [u8; 32]) -> Self {
        PlanBlueprint {
            events: Vec::new(),
            event_map: HashMap::new(),
            events_seen: HashMap::new(),
            stop_at: None,
            stop_exclude: Vec::new(),
            seed,
            workers: Vec::new(),
            worker_map: HashMap::new(),
        }
    }

    pub fn plan(self) -> Result<Plan, (Plan, Vec<Problem>)> {
        let planning = PlanningPhase::from(self);
        planning.plan()
    }

    pub fn stop_at(mut self, stop_at: f64) -> Self {
        self.stop_at = Some(stop_at);
        self
    }

    pub fn stop_exclude(mut self, stop_exclude: Vec<Uuid>) -> Self {
        self.stop_exclude = stop_exclude;
        self
    }

    pub fn event(mut self, plan: EventBlueprint) -> AnyResult<Self> {
        let id = EventId(self.events.len());
        #[cfg(feature = "tracing")]
        tracing::debug!(
            "New event {:?} has {:?} dependencies",
            plan.id,
            plan.dependencies
        );

        let assigned_worker = if let Some(assigned_worker) = &plan.assigned_worker {
            let mut worker_id = None;
            for (&worker_index, worker_uuid) in self.worker_map.iter() {
                if worker_uuid == assigned_worker {
                    worker_id = Some(worker_index);
                }
            }
            let worker_id = match worker_id {
                Some(worker_id) => worker_id,
                None => {
                    return Err(anyhow!("Worker {:?} not found", assigned_worker));
                }
            };
            let worker = self.workers.get_mut(worker_id.0).unwrap();

            if let Some(start) = plan.start {
                if !worker.is_available_for(Window {
                    start,
                    end: start + plan.duration,
                }) {
                    return Err(anyhow!(
                        "{:?} is not available at the given start time",
                        assigned_worker
                    ));
                }
            }
            Some(worker_id)
        } else {
            None
        };

        let mut dependencies = vec![];
        for connection in plan.dependencies {
            match self.events_seen.get(&connection.0) {
                Some(event_id) => {
                    dependencies.push(Connection(*event_id));
                }
                None => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        "Link connection {:?} -> {:?} not found",
                        plan.id,
                        connection.0
                    );
                }
            }
        }

        let event = Event::new(
            id,
            plan.start.unwrap_or(0.0),
            plan.duration,
            plan.priority,
            plan.attention,
            assigned_worker,
            plan.requirements,
            plan.constraints,
            dependencies,
            None,
        );
        self.event_map.insert(id, plan.id);
        self.events_seen.insert(plan.id, id);
        self.events.push(event);
        Ok(self)
    }

    pub fn worker(
        mut self,
        worker_id: Uuid,
        blocked_off: Constraints,
        capabilities: HashSet<Capability>,
    ) -> Self {
        let id = WorkerId(self.workers.len());
        let worker = Worker::new(id, blocked_off, capabilities);
        self.worker_map.insert(id, worker_id);
        self.workers.push(worker);
        self
    }
}
