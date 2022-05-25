import os
from hpobench.container.benchmarks.nas.nasbench_201 import Cifar100NasBench201Benchmark as Benchmark


container_source = os.path.join(os.getcwd(), 'containers/nas_benchmark')
print('container_source', container_source)
b = Benchmark(task_id=167149, container_source=container_source, rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
print('Config:', config)
print('Configspace', b.get_configuration_space())
result_dict = b.objective_function(configuration=config, rng=1)
print('Results:', result_dict)