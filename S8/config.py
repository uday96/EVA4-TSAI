import pprint

class ModelConfig(object):

	def __init__(self,):
		super(ModelConfig, self).__init__()
		self.seed = 1
		self.batch_size_cuda = 64
		self.batch_size_cpu = 64	
		self.num_workers = 4
		self.epochs = 50
		self.dropout = 0.15
		self.lr = 0.001
		self.l1_decay = 1e-6
		self.l2_decay = 1e-3

	def print_config(self):
		print("Model Parameters:")
		pprint.pprint(vars(self), indent=2)


def test_config():
	args = ModelConfig()
	args.print_config()

if __name__ == '__main__':
	test_config()
