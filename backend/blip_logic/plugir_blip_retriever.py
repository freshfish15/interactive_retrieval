# plugir_blip_retriever.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import entropy as scipy_entropy # Renamed to avoid conflict if any

class PlugIRStyleRetriever:
    def __init__(self, model_name_or_path="Salesforce/blip-itm-large-coco", device=None):
        """
        Initializes the retriever with BLIP models and processor.

        Args:
            model_name_or_path (str): The Hugging Face model name or path.
                                      Used for both ITM and potentially captioning.
            device (str, optional): The device to run the model on (e.g., "cuda", "cpu"). 
                                    Autodetects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"PlugIRStyleRetriever initializing on device: {self.device}")
        
        try:
            self.processor = BlipProcessor.from_pretrained(model_name_or_path)
            # Model for Image-Text Matching (Retrieval)
            self.retrieval_model = BlipForImageTextRetrieval.from_pretrained(model_name_or_path).to(self.device)
            self.retrieval_model.eval() 
            print(f"Successfully loaded BLIP retrieval model '{model_name_or_path}' (BlipForImageTextRetrieval) and processor.")

            # Model for Caption Generation
            # Note: If model_name_or_path is an ITM model, captioning quality might be suboptimal.
            # Consider using a dedicated captioning model like "Salesforce/blip-image-captioning-base"
            # if you pass that as model_name_or_path, or initialize this separately.
            self.caption_model = BlipForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
            self.caption_model.eval()
            print(f"Successfully loaded BLIP captioning model '{model_name_or_path}' (BlipForConditionalGeneration).")
            
            proj_dim = "N/A"
            if hasattr(self.retrieval_model, 'vision_proj') and hasattr(self.retrieval_model.vision_proj, 'out_features'):
                 proj_dim = self.retrieval_model.vision_proj.out_features
                 if hasattr(self.retrieval_model, 'text_proj') and hasattr(self.retrieval_model.text_proj, 'out_features'):
                     if self.retrieval_model.text_proj.out_features != proj_dim:
                         print(f"Warning: vision_proj.out_features ({proj_dim}) != text_proj.out_features ({self.retrieval_model.text_proj.out_features}). Using vision_proj dim.")
            elif hasattr(self.retrieval_model.config, 'projection_dim') and self.retrieval_model.config.projection_dim is not None:
                 proj_dim = self.retrieval_model.config.projection_dim 
            else: 
                 proj_dim = 256 # Fallback
                 print(f"Warning: Could not reliably determine projection_dim from model attributes. Defaulting to {proj_dim}. Verify this matches model's ITM head.")
            self.projection_dim = proj_dim
            print(f"Projected embedding dimension for ITM set to: {self.projection_dim}")

        except Exception as e:
            print(f"Error loading model or processor from '{model_name_or_path}': {e}")
            raise

    @torch.no_grad()
    def get_image_embeddings(self, images_pil_list, batch_size=32):
        """
        Computes L2-normalized, projected embeddings for a list of PIL images.
        Uses vision_model's last_hidden_state[:, 0] (CLS token) -> vision_proj.
        """
        if not images_pil_list:
            return np.array([])
        
        all_image_features_projected_normalized = []
        
        print(f"Computing projected image embeddings for {len(images_pil_list)} images...")
        for i in tqdm(range(0, len(images_pil_list), batch_size), desc="Image Embedding Batches"):
            batch_images_pil = images_pil_list[i:i + batch_size]
            try:
                inputs = self.processor(images=batch_images_pil, return_tensors="pt").to(self.device)
            except Exception as e:
                print(f"Error processing images for batch starting at index {i}: {e}"); continue

            vision_outputs = self.retrieval_model.vision_model(
                pixel_values=inputs.pixel_values, return_dict=True
            )
            image_cls_embedding = vision_outputs.last_hidden_state[:, 0, :] 
            image_embeds_projected = self.retrieval_model.vision_proj(image_cls_embedding)
            
            if image_embeds_projected is None: 
                print(f"Warning: Projected image embedding is None for batch {i}."); continue
            image_features_projected_normalized = F.normalize(image_embeds_projected, p=2, dim=-1)
            all_image_features_projected_normalized.append(image_features_projected_normalized.cpu().numpy())
        
        return np.concatenate(all_image_features_projected_normalized, axis=0) if all_image_features_projected_normalized else np.array([])

    @torch.no_grad()
    def get_text_embeddings(self, texts_list, batch_size=32):
        """
        Computes L2-normalized, projected embeddings for a list of text strings.
        Uses text_encoder's last_hidden_state[:, 0] (CLS token) -> text_proj.
        """
        if not texts_list:
            return np.array([])
            
        all_text_features_projected_normalized = []
        print(f"Computing projected text embeddings for {len(texts_list)} texts...")
        for i in tqdm(range(0, len(texts_list), batch_size), desc="Text Embedding Batches"):
            batch_texts = texts_list[i:i + batch_size]
            inputs = self.processor(
                text=batch_texts, return_tensors="pt", padding="max_length", 
                truncation=True, max_length=512
            ).to(self.device)
        
            text_encoder_module = None
            if hasattr(self.retrieval_model, 'text_encoder'):
                text_encoder_module = self.retrieval_model.text_encoder
            elif hasattr(self.retrieval_model, 'text_model'): 
                 text_encoder_module = self.retrieval_model.text_model
            else:
                print(f"Error: No text_encoder/text_model on self.retrieval_model for batch {i}"); continue 

            text_outputs = text_encoder_module(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True
            )
            text_cls_embedding = text_outputs.last_hidden_state[:, 0, :]
            text_embeds_projected = self.retrieval_model.text_proj(text_cls_embedding)

            if text_embeds_projected is None: 
                print(f"Warning: Projected text embedding is None for batch {i}."); continue
            
            text_features_projected_normalized = F.normalize(text_embeds_projected, p=2, dim=-1)
            all_text_features_projected_normalized.append(text_features_projected_normalized.cpu().numpy())

        return np.concatenate(all_text_features_projected_normalized, axis=0) if all_text_features_projected_normalized else np.array([])

    def retrieve_top_k_indices_and_scores(self, 
                                          query_embedding_normalized, 
                                          gallery_embeddings_normalized, 
                                          top_k=10):
        if query_embedding_normalized.ndim == 1:
            query_embedding_normalized = np.expand_dims(query_embedding_normalized, axis=0)
        if gallery_embeddings_normalized.shape[0] == 0: return np.array([]), np.array([])
        if query_embedding_normalized.shape[1] != gallery_embeddings_normalized.shape[1]:
            print(f"ERROR: Query dim {query_embedding_normalized.shape[1]} != Gallery dim {gallery_embeddings_normalized.shape[1]}.")
            return np.array([]), np.array([])
        similarities = np.dot(query_embedding_normalized, gallery_embeddings_normalized.T).squeeze()
        if similarities.ndim == 0: similarities = np.array([similarities.item()])
        num_gallery_items = gallery_embeddings_normalized.shape[0]
        actual_top_k = min(top_k, num_gallery_items)
        if actual_top_k == 0: return np.array([]), np.array([])
        if actual_top_k < num_gallery_items / 2 and actual_top_k > 0: 
            partitioned_indices = np.argpartition(similarities, -actual_top_k)[-actual_top_k:]
            scores_of_partitioned = similarities[partitioned_indices]
            sorted_indices_within_partition = np.argsort(scores_of_partitioned)[::-1] 
            top_k_indices = partitioned_indices[sorted_indices_within_partition]
            top_k_scores = similarities[top_k_indices]
        elif actual_top_k > 0 : 
            top_k_indices = np.argsort(similarities)[::-1][:actual_top_k] 
            top_k_scores = similarities[top_k_indices]
        else: return np.array([]), np.array([])
        return top_k_indices, top_k_scores

    def select_representative_images(self, candidate_ids, candidate_embeddings_normalized, num_representatives):
        """
        Selects representative images from candidates using K-means and entropy.
        Ref: PlugIR Algorithm 1 and Section 3.3 "Retrieval Context Extraction" [cite: 102, 105, 106, 107, 108, 109, 110]
        
        Args:
            candidate_ids (list): List of IDs for the candidate images.
            candidate_embeddings_normalized (np.ndarray): Normalized embeddings of candidate images (N, D).
            num_representatives (int): Number of representatives (m) to select.

        Returns:
            list: List of IDs of the selected representative images.
        """
        num_candidates = len(candidate_ids)
        if not candidate_ids or num_representatives <= 0 or num_candidates == 0:
            return []
        
        actual_m = min(num_representatives, num_candidates)
        if actual_m == num_candidates:
            return candidate_ids # Return all if m >= N

        # 1. Calculate pairwise similarities between all candidates
        # candidate_embeddings_normalized is (N, D)
        # sim_matrix will be (N, N)
        sim_matrix = np.dot(candidate_embeddings_normalized, candidate_embeddings_normalized.T)
        np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix) # Ensure values are in [-1, 1] for stability

        # 2. Calculate entropy for each candidate based on its similarity to OTHERS
        candidate_entropies = np.zeros(num_candidates)
        for i in range(num_candidates):
            # Similarities of image i to all OTHER candidate images
            similarities_to_others = np.delete(sim_matrix[i, :], i)
            
            if similarities_to_others.size == 0: # Should only happen if num_candidates = 1 (handled above)
                candidate_entropies[i] = float('inf') 
                continue

            # Convert similarities to a probability distribution using softmax
            # (as described in your original ImageContextExtractor, good for probabilities)
            exp_sim = np.exp(similarities_to_others - np.max(similarities_to_others)) # Stable softmax
            prob_dist = exp_sim / np.sum(exp_sim)
            prob_dist_cleaned = prob_dist[prob_dist > 1e-9] # Avoid log(0) for entropy if any prob is extremely small
            
            if prob_dist_cleaned.size == 0 or np.isclose(np.sum(prob_dist_cleaned), 0):
                candidate_entropies[i] = float('inf') # Assign high entropy if no valid prob_dist
            else:
                candidate_entropies[i] = scipy_entropy(prob_dist_cleaned, base=2)
        
        # 3. Perform K-means clustering
        try:
            # For scikit-learn >= 1.4, n_init='auto' is preferred. For older versions, an int like 10.
            # We'll try 'auto' and fallback if needed, or just use a default int.
            kmeans = KMeans(n_clusters=actual_m, random_state=42, n_init=10) 
            cluster_labels = kmeans.fit_predict(candidate_embeddings_normalized)
        except Exception as e:
            print(f"Error during KMeans clustering: {e}. Selecting top {actual_m} by lowest global entropy.")
            # Fallback: select m candidates with globally lowest entropy
            sorted_indices_by_entropy = np.argsort(candidate_entropies)
            return [candidate_ids[i] for i in sorted_indices_by_entropy[:actual_m]]

        # 4. Select one representative from each cluster (the one with the lowest entropy)
        representatives_ids = []
        selected_indices_global = set() # To avoid picking the same global index if clusters overlap in choices somehow

        for cluster_idx in range(actual_m):
            member_indices_in_candidates = [i for i, label in enumerate(cluster_labels) if label == cluster_idx]
            if not member_indices_in_candidates:
                # This can happen if a cluster is empty, e.g., k-means had issues or few points
                print(f"Warning: Cluster {cluster_idx} is empty.")
                continue

            min_entropy_in_cluster = float('inf')
            best_candidate_original_idx = -1

            for original_idx in member_indices_in_candidates:
                if candidate_entropies[original_idx] < min_entropy_in_cluster:
                    min_entropy_in_cluster = candidate_entropies[original_idx]
                    best_candidate_original_idx = original_idx
                # Optional: Tie-breaking (e.g., highest original similarity to query, or first one found)
            
            if best_candidate_original_idx != -1 and best_candidate_original_idx not in selected_indices_global:
                representatives_ids.append(candidate_ids[best_candidate_original_idx])
                selected_indices_global.add(best_candidate_original_idx)
            elif best_candidate_original_idx != -1:
                 # This candidate (by original_idx) was already chosen for another cluster's min entropy
                 # This should ideally not happen if each point belongs to one cluster.
                 # If it can, we might need to pick the next best in this cluster not already selected.
                 # For now, we just note it or allow duplicates if ID list allows. If IDs are unique, this is fine.
                 print(f"Warning: Candidate {candidate_ids[best_candidate_original_idx]} for cluster {cluster_idx} was already selected.")


        # Ensure we have `actual_m` representatives if possible, fill if some clusters were empty or problematic
        if len(representatives_ids) < actual_m and len(representatives_ids) < num_candidates:
            print(f"Selected {len(representatives_ids)} reps, needed {actual_m}. Filling with lowest entropy overall.")
            remaining_candidates_info = []
            for i in range(num_candidates):
                if candidate_ids[i] not in representatives_ids:
                    remaining_candidates_info.append({'id': candidate_ids[i], 'entropy': candidate_entropies[i]})
            
            remaining_candidates_info.sort(key=lambda x: x['entropy'])
            
            needed_more = actual_m - len(representatives_ids)
            for i in range(min(needed_more, len(remaining_candidates_info))):
                representatives_ids.append(remaining_candidates_info[i]['id'])
        
        return representatives_ids

    @torch.no_grad()
    def generate_captions(self, images_pil_list, batch_size=8, **generate_kwargs):
        """
        Generates captions for a list of PIL images.

        Args:
            images_pil_list (list of PIL.Image.Image): Images to caption.
            batch_size (int): Batch size for caption generation.
            **generate_kwargs: Additional arguments for caption_model.generate()
                               (e.g., max_length, num_beams).

        Returns:
            list: A list of generated caption strings.
        """
        if not images_pil_list:
            return []

        # Default generation parameters if not provided
        default_kwargs = {'max_length': 74, 'num_beams': 5, 'early_stopping': True}
        final_generate_kwargs = {**default_kwargs, **generate_kwargs}

        all_captions = []
        print(f"Generating captions for {len(images_pil_list)} images...")
        for i in tqdm(range(0, len(images_pil_list), batch_size), desc="Captioning Batches"):
            batch_images_pil = images_pil_list[i:i + batch_size]
            try:
                inputs = self.processor(images=batch_images_pil, return_tensors="pt").to(self.device)
            except Exception as e:
                print(f"Error processing images for captioning batch {i}: {e}")
                all_captions.extend(["Error generating caption"] * len(batch_images_pil))
                continue
            
            generated_ids = self.caption_model.generate(
                pixel_values=inputs.pixel_values, # Correct argument name
                **final_generate_kwargs
            )
            batch_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_captions.extend([caption.strip() for caption in batch_captions])
            
        return all_captions


if __name__ == '__main__':
    print("Testing PlugIRStyleRetriever with BlipForImageTextRetrieval and Captioning...")
    # Use a model good for both ITM and captioning if possible, or be mindful of the choice.
    # "Salesforce/blip-image-captioning-large" is good for captioning and its vision/text parts can be used for ITM.
    # "Salesforce/blip-itm-large-coco" is good for ITM; captioning might be okay.
    retriever = PlugIRStyleRetriever(model_name_or_path="Salesforce/blip-image-captioning-base") # Changed for better captioning example

    try:
        image_size = 384 
        if hasattr(retriever.processor, 'image_processor') and \
           hasattr(retriever.processor.image_processor, 'size') and \
           'height' in retriever.processor.image_processor.size:
            image_size = retriever.processor.image_processor.size['height']
        print(f"Using image size: {image_size} for dummy images.")

        dummy_image_red = Image.new('RGB', (image_size, image_size), color = 'red') 
        dummy_image_blue = Image.new('RGB', (image_size, image_size), color = 'blue')
        dummy_image_green = Image.new('RGB', (image_size, image_size), color = 'green')
        dummy_image_yellow = Image.new('RGB', (image_size, image_size), color = 'yellow')
        
        candidate_images_pil = [dummy_image_red, dummy_image_blue, dummy_image_green, dummy_image_yellow]
        candidate_ids = ["red_img", "blue_img", "green_img", "yellow_img"]
        
        print("\n--- Testing Image Embeddings ---")
        candidate_embeds = retriever.get_image_embeddings(candidate_images_pil, batch_size=2)
        print(f"Candidate embeddings shape: {candidate_embeds.shape}") 

        if candidate_embeds.shape[0] > 0:
            print("\n--- Testing Representative Image Selection ---")
            num_reps = 2
            representative_img_ids = retriever.select_representative_images(
                candidate_ids, candidate_embeds, num_representatives=num_reps
            )
            print(f"Selected {len(representative_img_ids)} representative IDs: {representative_img_ids}")

            # For captioning, we'd need the actual PIL images for these IDs
            # This example assumes we have them or can retrieve them.
            # Here, we'll just try to caption the first few candidates for demonstration.
            if representative_img_ids:
                # Create a map for easy lookup
                id_to_pil_map = {id_val: pil_img for id_val, pil_img in zip(candidate_ids, candidate_images_pil)}
                imgs_to_caption = [id_to_pil_map[id_val] for id_val in representative_img_ids if id_val in id_to_pil_map]
                
                if imgs_to_caption:
                    print(f"\n--- Testing Caption Generation for {len(imgs_to_caption)} representatives ---")
                    captions = retriever.generate_captions(imgs_to_caption, batch_size=2)
                    for img_id, caption in zip(representative_img_ids, captions):
                        print(f"  ID: {img_id}, Generated Caption: '{caption}'")
                else:
                    print("No representative images found to caption based on selected IDs.")
            else:
                print("No representative images selected to test captioning.")

    except ImportError: print("Pillow (PIL) or other dependencies not installed.")
    except Exception as e: print(f"An error occurred: {e}")

