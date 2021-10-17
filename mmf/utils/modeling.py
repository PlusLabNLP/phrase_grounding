# Copyright (c) Facebook, Inc. and its affiliates.

from torch import nn


ACT2FN = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def get_bert_configured_parameters(module, lr=None, param_optimizer=None):
    if param_optimizer is None:
        param_optimizer = list(module.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if lr is not None:
        for p in optimizer_grouped_parameters:
            p["lr"] = lr

    return optimizer_grouped_parameters


def get_optimizer_parameters_for_bert(module, config):
    # Pretraining has same LR for all of the parts
    if module.config.training_head_type == "pretraining":
        base_lr_multiplier = getattr(config.model_config, "base_lr_multiplier", 1)
        if base_lr_multiplier != 1:
            encoder = module.bert
            encoder_param_optimizer = list(encoder.named_parameters())
            generator = module.generator
            generator_param_optimizer = list(generator.named_parameters())
            generator_param_optimizer = [x for x in generator_param_optimizer if not x in encoder_param_optimizer and not 'bert.embeddings.word_embeddings' in x[0] and not 'bert.embeddings.projection' in x[0] and not 'generator_lm_head' in x[0]]
            discriminator_predictions = module.discriminator_predictions
            pred_param_optimizer = list(discriminator_predictions.named_parameters())
            pred_param_optimizer = [x for x in pred_param_optimizer if not x in encoder_param_optimizer and not x in generator_param_optimizer]
            lr = config.optimizer.params.lr
            parameters = get_bert_configured_parameters(encoder, lr*base_lr_multiplier, encoder_param_optimizer)
            parameters += get_bert_configured_parameters(generator, lr, generator_param_optimizer)
            parameters += get_bert_configured_parameters(discriminator_predictions, lr, pred_param_optimizer)
            return parameters


        return get_bert_configured_parameters(module)

    # For finetuning setup, we have classifier
    lr = config.optimizer.params.lr
    model_config = getattr(config.model_config, config.model, {})
    finetune_lr_multiplier = getattr(model_config, "finetune_lr_multiplier", 1)
    # Finetune the bert pretrained part with finetune_lr_multiplier if it is set
    encoder = module.transformer if hasattr(module, "transformer") else module.bert
    parameters = get_bert_configured_parameters(encoder, lr * finetune_lr_multiplier)
    # Classifier will be trained on the normal lr
    parameters += get_bert_configured_parameters(module.classifier, lr)

    return parameters
