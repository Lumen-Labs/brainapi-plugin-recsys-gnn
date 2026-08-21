from kombu import Queue

QUEUES = (Queue("recsys_gnn", routing_key="recsys_gnn"),)
ROUTES = {
    "workers.tasks.train_recsys": {"queue": "recsys_gnn"},
}
