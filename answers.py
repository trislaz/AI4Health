# Question 1.1
def get_magnification(level):
    assert level < wsi.level_count, "Level is too high, the pyramidal object does not contain it."
    return float(wsi.properties['openslide.objective-power']) / wsi.level_downsamples[level]

center_x = dimensions[0] // 2
center_y = dimensions[1] // 2
region_size = (256, 256)
region_high = wsi.read_region(
    (center_x - region_size[0]//2, center_y - region_size[1]//2),
    0,
    region_size
)

region_mid = wsi.read_region(
    (center_x, center_y),
    1,
    region_size
)

region_low = wsi.read_region(
    (center_x, center_y),
    2,
    region_size
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(region_high)
axes[0].set_title(f'Level 0 (Resolution: {get_magnification(0):.1f}x)')
axes[0].axis('off')
axes[1].imshow(region_mid)
axes[1].set_title(f'Level 1 (Resolution: {get_magnification(1):.1f}x)')
axes[1].axis('off')
axes[2].imshow(region_low)
axes[2].set_title(f'Level 2 (Resolution: {get_magnification(2):.1f}x)')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# Quesion 1.2
print(f"The total number of pixels in the slide is {math.prod(wsi.level_dimensions[0]):.2e}")

# Question 1.3
level_tile = int(math.log(mag_level0 / magnification_tile, ds_per_level[ext]))
console.print(Panel(f"Tiling Configuration:\n• Target magnification: {magnification_tile}x\n• Pyramid level to use: {level_tile}\n• Tile size: {tile_size}x{tile_size} pixels\n• Tissue threshold: {mask_tolerance*100:.0f}%", title="Tiling Setup"))

# Question 1.4
def grid_coords(slide, point_start, point_end, patch_size):
    """Generate uniform grid coordinates."""
    size_x, size_y = patch_size
    list_col = range(point_start[1], point_end[1], size_x)
    list_row = range(point_start[0], point_end[0], size_y)
    return list(itertools.product(list_row, list_col))

# Question 1.5
def check_coordinates(row, col, patch_size, mask, mask_downsample, mask_tolerance=0.9):
    """Check if a tile location has sufficient tissue content."""
    col_0, row_0 = col, row
    col_1, row_1 = col + patch_size[0], row + patch_size[1]
    
    # Convert coordinates to mask resolution
    col_0, row_0 = col_0 // mask_downsample, row_0 // mask_downsample
    col_1, row_1 = col_1 // mask_downsample, row_1 // mask_downsample
    
    # Check bounds
    if col_0 < 0 or row_0 < 0 or row_1 > mask.shape[0] or col_1 > mask.shape[1]:
        return False
    
    # Extract mask patch and check tissue content
    mask_patch = mask[row_0:row_1, col_0:col_1]
    tissue_ratio = mask_patch.sum() / mask_patch.size
    
    return tissue_ratio >= mask_tolerance

# Question 1.6
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=10, random_state=7)
kmeans.fit(embeddings)

tsne = TSNE(n_components=2, random_state=7)
embeddings_tsne = tsne.fit_transform(embeddings)
labels = kmeans.labels_

# Let's visualize the clusters
for i in range(10):
    plt.scatter(embeddings_tsne[labels == i, 0], embeddings_tsne[labels == i, 1], label=f'Cluster {i}')
plt.legend()
plt.show()

## Visualization of the clusters
def extract_from_cluster(cluster_id, sqrt_n_tiles=3, coordinates=coordinates, labels=labels, wsi=wsi):
    """
    Extracts n_tiles from the cluster_id cluster.
    """
    tiles = random.sample([tile for o, tile in enumerate(coordinates) if labels[o] == cluster_id], sqrt_n_tiles**2)
    images = []
    for tile in tiles:
        images.append(np.array(extract_tile(wsi, tile[0], tile[1], 0, 256).convert("RGB")))
    images = einops.rearrange(images, "(n1 n2) h w c -> (n1 h) (n2 w) c", n1=sqrt_n_tiles, n2=sqrt_n_tiles)
    return Image.fromarray(images)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(extract_from_cluster(9))
axes[1].imshow(extract_from_cluster(6))

# Question 2.1
"""
- **For the image**: it is automatically reshaped, cropped etc... → Be careful with what you feed in; this might remove important information.
- **For the text**: Try to tokenize several sentences. All token sequences are of size 77. They all start and end with a given token id - SOS and EOS.

Indeed in the CLIP model, text encoder was using absolute positional embeddings — which fixed the maximum sequence to 77. 
In addition, contrary to other models such as BERT, CLIP encoder was thought as a decoder only model (like GPTs are) and we use the EOS token as the encoding for the full sentence.
"""


# Question 2.2
def cosine(x, y):
    """Compute cosine similarity between two tensors after normalization."""
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    return x @ y.T

# You can solve this kind of "easy" classification tasks
texts = ["An image of a map", "An image of a volcano"]
texts_tokens = tokenizer(texts)
texts_encoded = clip_model.encode_text(texts_tokens)

# Compute the cosine similarity between the image and text embeddings
brit_similarity = cosine(brit_encoded, texts_encoded)
it_similarity = cosine(it_encoded, texts_encoded)

table = Table(title="Cosine Similarity")
table.add_column("Image")
table.add_column("Map", justify="right")
table.add_column("Volcano", justify="right")
table.add_row("Brittany", f"{brit_similarity[0][0].item():.3f}", f"{brit_similarity[0][1].item():.3f}")
table.add_row("Italy", f"{it_similarity[0][0].item():.3f}", f"{it_similarity[0][1].item():.3f}")

console.print(table)

# Question 2.4

# Let's define the text prompts.
text_prompts = ["A histopathological tile with tumour cells", "An histopathological tile without tumour cells"]
text_clip = tokenizer(text_prompts)

tumor_imgs = [Image.open(p) for p in tumor_tiles]
stroma_imgs = [Image.open(p) for p in stroma_tiles]

tumor_img_clip = torch.stack([preprocess_clip(img) for img in tumor_imgs])
stroma_img_clip = torch.stack([preprocess_clip(img) for img in stroma_imgs])

with torch.no_grad():
    tumor_embeddings = clip_model.encode_image(tumor_img_clip)
    stroma_embeddings = clip_model.encode_image(stroma_img_clip)
    text_embeddings = clip_model.encode_text(text_clip)

tumor_logits = cosine(tumor_embeddings, text_embeddings)
stroma_logits = cosine(stroma_embeddings, text_embeddings)

table = Table(title="CLIP Classification Logits")
table.add_column("Image Sample", justify="left", style="cyan", no_wrap=True)
table.add_column(f"Logit: '{text_prompts[0]}'", justify="right")
table.add_column(f"Logit: '{text_prompts[1]}'", justify="right")

tumor_probs = tumor_logits.softmax(dim=-1)
stroma_probs = stroma_logits.softmax(dim=-1)

for i, (logits, p) in enumerate(zip(tumor_probs, tumor_tiles)):
    l1, l2 = logits
    sample_name = f"tumor_{i} ({Path(p).name})"
    table.add_row(sample_name, f"{l1:.4f}", f"{l2:.4f}")

table.add_section()
for i, (logits, p) in enumerate(zip(stroma_probs, stroma_tiles)):
    l1, l2 = logits
    sample_name = f"stroma_{i} ({Path(p).name})"
    table.add_row(sample_name, f"{l1:.4f}", f"{l2:.4f}")

console.print(table)

# Question 2.6

"""
As you can see, captions are very detailed but images quality may vary a lot:
- Magnification is not consistent
- There are numerous artifacts, texts, pen strokes, etc.
- This is indeed targeting a human audience (therefore, the pen marks etc...)
- Many image components are not annotated because too obvious (e.g., presence of lymphocytes in the right image)
"""

# Question 2.7

tiles_embeddings = torch.vstack([tumor_embeddings, stroma_embeddings])
tiles_path = [Path(p) for p in tumor_tiles + stroma_tiles]
text_probe = tokenizer(["An H&E image with tumor cells"])
# An H&E image with a low density of cells

with torch.no_grad():
    text_probe_embeddings = clip_model.encode_text(text_probe)

similarity = cosine(tiles_embeddings, text_probe_embeddings).squeeze()
sorted_indices = torch.argsort(similarity, descending=True)

best_idx, worst_idx = sorted_indices[0], sorted_indices[-1]
best_score, worst_score = similarity[best_idx].item(), similarity[worst_idx].item()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(Image.open(tiles_path[best_idx]))
axes[0].set_title(f'Best match\nScore: {best_score:.3f}', 
                  color='green')
axes[0].axis('off')
axes[1].imshow(Image.open(tiles_path[worst_idx]))
axes[1].set_title(f'Worst match\nScore: {worst_score:.3f}', 
                  color='red')
axes[1].axis('off')

plt.tight_layout()
plt.show()

top_k = 4
descending = True 
indices_to_show = sorted_indices[-top_k:] if not descending else sorted_indices[:top_k]
scores_to_show = similarity[indices_to_show].squeeze()

fig, axes = plt.subplots(1, top_k, figsize=(15, 4))
for i, (idx, score) in enumerate(zip(indices_to_show, scores_to_show)):
    axes[i].imshow(Image.open(tiles_path[idx]))
    axes[i].set_title(f'Rank {len(tiles_path)-top_k+i+1 if not descending else i+1}\nScore: {score:.3f}', 
                     color='red' if not descending else 'green')
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# Question 2.8


def get_description_prompt(caption):
    prompt = f"""
    You are a histopathologic expert. Given an H&E image, your task is to extend an incomplete image description. Focus on the details you can see in the image.

    Incomplete description: {caption}
    """
    return prompt

# If you want help creating a good prompt, there are online tools that do just that.
# For instance, anthropic proposes this service (but you would need to have an API key).
# Here is the prompt he created given mine:

def get_description_prompt_anthropic(caption):
    """
    Creates a detailed prompt for a vision model, based on Anthropic's template,
    to extend a histopathology image caption.
    Your answer will start with the original incomplete description, then seamlessly integrate your additional observations and details.
    """
    prompt = f"""You are a histopathologic expert tasked with extending an incomplete description of an H&E (Hematoxylin and Eosin) stained tissue image. 
    Your goal is to provide a detailed and accurate description of the image, focusing on the histological features and any potential pathological findings.

First, you are presented with an incomplete description of the H&E image:

Incomplete description:
{caption}

Next, you will be shown the H&E image.

Carefully examine the H&E image, paying attention to the following aspects:
1. Tissue type and architecture
2. Cellular composition and distribution
3. Nuclear characteristics (size, shape, chromatin pattern)
4. Cytoplasmic features
5. Presence of any abnormal structures or cellular arrangements
6. Stromal components
7. Vascular elements
8. Any signs of inflammation, necrosis, or other pathological processes

Based on your observations, extend the incomplete description by adding relevant details about the histological features you can see in the image. Be specific and use appropriate medical terminology. If you notice any potential pathological findings, describe them objectively without making a definitive diagnosis.

Start with the original incomplete description, then seamlessly integrate your additional observations and details.]

Remember to maintain a professional and objective tone throughout your description. Focus on what you can directly observe in the image, and avoid speculating about diagnoses or clinical implications unless they are clearly evident from the histological features.
Answer ONLY with your extended description, nothing else."""
    return prompt

client = SimpleChatClient("https://trizard-ai4health.hf.space/chat")

for item in captions:
    image_path = Path('assets') / item['filename']
    
    new_caption = client.chat(get_description_prompt_anthropic(item['caption']), image_path=image_path)
    print(new_caption)
    
    panel_content = (
        f"[bold]Original Caption:[/bold]\n{item['caption']}\n\n"
        f"[bold cyan]Augmented Caption:[/bold cyan]\n{new_caption}"
    )
    img = Image.open(Path('assets') / item['filename'])
    console.print(Panel(panel_content, title=f"Caption Augmentation for {item['filename']}", border_style="blue"))


# Question 2.10

prompts = {
    "addition": lambda x: f"You are tasked to selectively degrade an image description by adding a the description of a new visual detail (and only that!). You will answer only with the new, degraded description. \n\n Original description: {x}",
    "deletion": lambda x: f"You are tasked to selectively degrade an image description by removing the description of a visual detail from it (and only that!). You will answer only with the new, degraded description. \n\n Original description: {x}",
    "modification": lambda x: f"You are tasked to selectively degrade an image description by modifying the description of a visual detail from it (and only that!) - modify it such that it mainly changes the meaning of its description. You will answer only with the new, degraded description. \n\n Original description: {x}",
}

def degrade_description(image_description):
    # Randomly select a degradation type
    deg_type = random.choice(list(prompts.keys()))
    prompt = prompts[deg_type](image_description)

    completion = client.chat(prompt, image_path=image_path)
    return deg_type, completion

deg_type, degraded_description = degrade_description(image_description)

# Question 2.12

# Using GPT generated prompts:
prompts_from_gpt = [client.chat("You are an expert in histopathology. You are task to describe one pattern that could be present in a breast cancer slide. Answer exclusively by giving an example pattern.", max_tokens=10) for _ in range(3)]
tokenized_prompts = tokenizer(prompts_from_gpt)
prompts_gpt_embeddings = clip_model.encode_text(tokenized_prompts)

prompts = torch.vstack([prompts_report_embeddings, prompts_gpt_embeddings])

# Question 2.13

# They augmented their tile dataset with a KMeans approach.
# Each cluster being sampled uniformly.
# Calculate number of clusters - square root of total patches
n_clusters = int(math.sqrt(len(tiles)))
print(f"Using {n_clusters} clusters for {len(tiles)} total tiles")

kmeans = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = kmeans.fit_predict(tiles)

samples_per_cluster = 256 // n_clusters
remaining_samples = 256 % n_clusters

cluster_sampled_indices = []
for cluster_idx in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster_idx)[0]
    # Add one extra sample to some clusters if we have remaining samples
    n_samples = samples_per_cluster + (1 if cluster_idx < remaining_samples else 0)
    # We sample randomly from the cluster
    if len(cluster_indices) > n_samples:
        sampled_indices = np.random.choice(cluster_indices, size=n_samples, replace=False)
        cluster_sampled_indices.extend(sampled_indices)
    else:
        # This is highly unlikely at 10 or 20x
        cluster_sampled_indices.extend(cluster_indices)

cluster_sampled_indices = torch.tensor(cluster_sampled_indices)
cluster_extracted_tiles = tiles[cluster_sampled_indices]

# Combine with prompt-based extracted tiles (which we got earlier)
# Remove any duplicates that might exist between the two methods
combined_indices = torch.concat([first_128_indices, cluster_sampled_indices])
unique_indices = torch.unique(combined_indices)

# # Get final selection of tiles
kmeans_extracted_tiles = tiles[unique_indices]
final_tiles = torch.concat([torch.tensor(prompt_extracted_tiles), torch.tensor(kmeans_extracted_tiles)])
print(f"Final number of extracted tiles: {len(final_tiles)}")

# Question 2.14

# Let's define the text prompts.
text_prompts = ["A histopathological tile with tumour cells", "An histopathological tile without tumour cells"]
text_clip = tokenizer(text_prompts)

tumor_imgs = [Image.open(p) for p in tumor_tiles]
stroma_imgs = [Image.open(p) for p in stroma_tiles]

tumor_img_clip = torch.stack([preprocess_clip(img) for img in tumor_imgs])
stroma_img_clip = torch.stack([preprocess_clip(img) for img in stroma_imgs])

# Get the embeddings
with torch.no_grad():
    tumor_embeddings = clip_model.encode_image(tumor_img_clip)
    stroma_embeddings = clip_model.encode_image(stroma_img_clip)
    text_embeddings = clip_model.encode_text(text_clip)

# Compute logits using our simple cosine function
tumor_logits = cosine(tumor_embeddings, text_embeddings)
stroma_logits = cosine(stroma_embeddings, text_embeddings)

table = Table(title="CLIP Classification Logits")
table.add_column("Image Sample", justify="left", style="cyan", no_wrap=True)
table.add_column(f"Logit: '{text_prompts[0]}'", justify="right")
table.add_column(f"Logit: '{text_prompts[1]}'", justify="right")

tumor_probs = tumor_logits.softmax(dim=-1)
stroma_probs = stroma_logits.softmax(dim=-1)

for i, (logits, p) in enumerate(zip(tumor_probs, tumor_tiles)):
    l1, l2 = logits
    sample_name = f"tumor_{i} ({Path(p).name})"
    table.add_row(sample_name, f"{l1:.4f}", f"{l2:.4f}")

table.add_section()
for i, (logits, p) in enumerate(zip(stroma_probs, stroma_tiles)):
    l1, l2 = logits
    sample_name = f"stroma_{i} ({Path(p).name})"
    table.add_row(sample_name, f"{l1:.4f}", f"{l2:.4f}")

console.print(table)

tumor_preds = tumor_probs.argmax(dim=-1)
stroma_preds = stroma_probs.argmax(dim=-1)

tumor_accuracy = (tumor_preds == 0).float().mean()
stroma_accuracy = (stroma_preds == 1).float().mean()

console.print(f"Tumor tile classification accuracy: {tumor_accuracy.item():.2f}")
console.print(f"Stroma tile classification accuracy: {stroma_accuracy.item():.2f}")
console.print(f"Overall classification accuracy: {(tumor_accuracy + stroma_accuracy) / 2:.2f}")
