## tensorflow v1 框架优化器文档：

### One sparse optimizers are released

* MomentumOptimizerSparse (pruning functions added into official SGD optimizer)

Besides the common parameters used in optimizers, such as learning rate, momentum, the sparse optimizers have a few pruning-related parameters, as listed below:

* init_sparsity=0 (float, default=0):

      initial sparsity

* target_sparsity (float, default=0):

      final sparsity, e.g. 0.95 

* pretrain_step (int, default=0):

      pretrained steps before pruning. This value should be set to 0 if a pretrained model is loaded

* sparse_step (int, default=0):

      pruning within the sparse step 

* frequency (int, default=100):

      in how many steps, pruning a interval toward the target sparsity

* keywords_no_sparse (list, default=['bias', 'beta', 'gamma']):

      parameters to be excluded during pruning

* special_sparsity_dict (dict, default={}):

      name-sparsity pairs with target sparsity different from the default one, such as special_sparsity_dict={'layer1.weight':0.2}

* enable_channel_balance (bool, default=True):

      bank balanced setup based on hardware resources

* finetune (bool, default=False):

      restore the sparsity in the pretrained model and finetune the sparse network on new datasets