_base_ = ['./base_e2e.py'] # 继承你最原始的配置文件
model = dict(planning_head=dict(
    enable_ood_injection=False # 彻底关闭 OOD 注入，退化为纯 UniAD
))
