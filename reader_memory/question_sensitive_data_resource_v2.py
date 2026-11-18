"""Resource-only revision before any pilot output; original worker stays frozen."""
import traceback
from . import question_sensitive_data_v1 as worker

worker.CONFIG = dict(worker.CONFIG, allocator_cap_mib=10240)
worker.BASE = worker.BASE / 'resource_v2'
worker.RUN = worker.ROOT / 'runs/question_sensitive_data_pilot_resource_v2'

if __name__ == '__main__':
    try:
        worker.run()
    except Exception as exc:
        worker.RUN.mkdir(parents=True, exist_ok=True)
        worker.state('failed', error=repr(exc), traceback=traceback.format_exc())
        raise
