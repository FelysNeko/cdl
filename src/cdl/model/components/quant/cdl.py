from cdl.api.quant import QTensor, Quant
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


class CdlTensor(QTensor): ...


class CdlQuant(Quant): ...


class CdlQuantForActivation(CdlQuant): ...


class CdlQuantForWeight(CdlQuant): ...


class CdlMixin(BatchNormBatchStatsMixin): ...
