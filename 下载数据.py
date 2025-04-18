from modelscope.msdatasets import MsDataset
ds = MsDataset.load('gongjy/minimind-v_dataset', subset_name='default', split='test')
#您可按需配置 subset_name、split，参照“快速使用”示例代码