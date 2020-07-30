## pytorch sparse optimizer：

### Two sparse optimizers are released

* SGDSparse (pruning functions added into official SGD optimizer)

* AdamSparse (pruning functions added into official Adam optimizer)

Besides the common parameters used in optimizers, such as learning rate, momentum, the sparse optimizers have a few pruning-related parameters, as listed below:
* target_sparsity (float, default=0):

      final sparsity, e.g. 0.95 

* pretrain_step (int, default=0):

      pretrained steps before pruning. This value should be set to 0 if a pretrained model is loaded

* sparse_step (int, default=0):

      pruning steps 

* frequency (int, default=100):

      pruning frequency  

* keywords_no_sparse (list, default=['bias', 'beta', 'gamma']):

      parameters to be excluded during pruning

* special_sparsity_dict (dict, default={}):

      name-sparsity pairs with target sparsity different from the default one, such as special_sparsity_dict={'layer1.weight':0.2}

* sparse_block (dict, default={'1d': [1], '2d': [1, 1], '4d': [-1, 4, -1, 1]}):

      bank balanced setup based on hardware resources

* restore_sparsity (bool, default=False):

      restore the sparsity in the pretrained model. It is used when finetune sparse network on new datasets

* fix_sparsity (bool, default=False):

      fix the sparsity during training. It is generally used when finetune sparse network on new datasets

* L1_regularization (bool, default=False):

      whether to use l1 regularization. We found that l1 regularization may damage accuracy during pruning, therefore we do not recommend it

* param_name (iterator):

      pytorch specific, need to be passed as 
      param_name=model.named_parameters()