#%%
import datasets
import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import CLIPVisionModel

from infrastructure.dataset import DATASETS
from infrastructure.settings import DEVICE, DTYPE


if __name__ == "__main__":
    # M = nn.Parameter(torch.randn((1000, 5)))
    # # V, L = NCUT(num_eig=10, device=DEVICE).fit_transform(M)
    # # print(V.shape, L.shape)
    # # print(L, V[:, 0])
    # indices = torch_cluster.fps(M, batch=torch.arange(1000), ratio=0.25)
    # # print(indices, indices.shape)
    # raise Exception()

    dataset_name, n_classes = DATASETS["Common"][1]
    base_model_name = "facebook/dino-vitb8"

    # SECTION: Dataset setup
    dataset = datasets.load_dataset(dataset_name)
    dataset_size = dataset["train"].num_rows
    images = [sample["image"] for sample in (*dataset["train"],)]
    
    print(len(images))
    raise Exception()

    def process_grayscale(im):
        arr = np.array(im)
        return arr if arr.ndim >= 3 else np.tile(arr[..., None], (1, 1, 3))
    images = [*map(process_grayscale, images)]

    # SECTION: Debugging
    from transformers import ViTModel, ViTImageProcessor
    from model.multistate_encoder.modeling_msvitencoder import MultiStateViTConfig, MultiStateViTEncoderModel
    from model.clustering.modeling_fps import FPSClusteringConfig
    
    torch.manual_seed(1212)
    torch.cuda.empty_cache()
    base = ViTModel.from_pretrained(base_model_name)

    image_size = 224
    image_processor = ViTImageProcessor.from_pretrained(base_model_name)
    image_processor.__dict__.update({
        "size": {"height": image_size, "width": image_size},
    })
    inputs = image_processor(images=images, return_tensors="pt")

    model = MultiStateViTEncoderModel(MultiStateViTConfig(
        **base.config.to_dict(),
        _attn_implementation="eager",
        pregeneration_period=10,
        generation_period=2,
        clustering_config=FPSClusteringConfig(
            ncut_dim=100,
            fps_dim=8,
            fps_sample1=300,
            fps_sample2=100,
            fps_supersample2=120,
            cosine_similarity_threshold=0.7,
        ),
        pretrained=base_model_name
    ))
    print(model)
    print(model.config)
    # for image in images[:3]:
    #     plt.imshow(image)
    #     plt.show()
    with torch.no_grad():
        print(model(**inputs, interpolate_pos_encoding=True))
    raise Exception()

    # SECTION: Model setup
    model = CLIPVisionModel.from_pretrained(model_name)
    # print(model)

    print(model)
    print(model.config)

    # model.embeddings(**inputs)
    a = model.get_input_embeddings()(inputs["pixel_values"])
    print(a.dtype, a.shape)

    print()
    print(model.vision_model.embeddings(inputs["pixel_values"]).shape)
    raise Exception()

    affinity_focal_gamma = 1.0
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for layer, states in enumerate(hidden_states):
            X = states[..., 1:, :].flatten(0, -2)

            normalized_X = torch.nn.functional.normalize(X, dim=-1)
            normalized_A = 1.0 - normalized_X @ normalized_X.mT
            A = (X.norm(dim=-1)[:, None] * X.norm(dim=-1)[None, :]) * normalized_A

            A = torch.exp(-A / affinity_focal_gamma)

            D = A.sum(dim=-1)
            L = torch.eye(len(D)) - A * ((D[:, None] * D[None, :]) ** -0.5)

            E, V = torch.linalg.eigh(L)
            X = V[:, :10]

            X_embedded = TSNE(n_components=2).fit_transform(X)

            plt.scatter(*X_embedded.T)
            plt.title(f"Layer {layer}")
            plt.show()

            print(layer, states.shape, X_embedded.shape)






# %%