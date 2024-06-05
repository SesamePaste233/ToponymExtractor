from StyleEncoder.DeepFont import DeepFontEncoder, EncodeFontBatch, load_model
from Utils import bezier_utils as butils

def generate_style_embeddings(results, original_image, deepfont_encoder:DeepFontEncoder):
    snippets = []
    for r in results:
        snippet, _ = butils.get_bezier_bbox(original_image, r['upper_bezier_pts'], r['lower_bezier_pts'], scale=1.1)
        snippets.append(snippet)

    embeddings = EncodeFontBatch(deepfont_encoder, snippets)

    for r, embedding in zip(results, embeddings):
        r['style_embedding'] = embedding
    
    return results