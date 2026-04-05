from typing import Any, Tuple
from transformers import Trainer, TrainingArguments

def get_paged_adamw_32bit_manual(trainer: Trainer):
    """ 
    Note: you have to partially instantiate the trainer object to get the optimizer and then re init it with the optimizer you just created 
    (and scheduler if you have a scheduler you want to init manually).
    
    ref: https://discuss.huggingface.co/t/how-do-you-manually-create-a-paged-optimizer-32-bit-object-in-hf/70314/2 
    """
    # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args)
    # calls optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args) uses bits and bytes to get the paged optimizer 
    optimizer = trainer.create_optimizer()
    return optimizer 