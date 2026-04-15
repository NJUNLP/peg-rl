from comet import download_model, load_from_checkpoint
from verl.workers.comet.npu import NPUAccelerator
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from modelscope.utils.hf_util import patch_hub

patch_hub()
# Choose your model from Hugging Face Hub
# or for example:
# model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint("xx").to("npu")

# Data must be in the following format:
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    }
]
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1, accelerator=NPUAccelerator(), strategy=SingleDeviceStrategy(device="npu"))