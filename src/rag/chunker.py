def split_text(text, max_chunk_size=600, overlap=100):
    """
    Recursively split text into chunks based on paragraph, sentence, and word boundaries.
    """
    if not text:
        return []
        
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _split(txt, seps):
        if len(txt) <= max_chunk_size:
            return [txt]
            
        if not seps:
            # Hard limit fallback splitting with overlap
            chunks = []
            start = 0
            while start < len(txt):
                end = min(start + max_chunk_size, len(txt))
                chunks.append(txt[start:end])
                if end == len(txt):
                    break
                start += (max_chunk_size - overlap)
            return chunks
            
        sep = seps[0]
        # Handle cases where the separator is not found
        if sep not in txt:
            return _split(txt, seps[1:])
            
        parts = txt.split(sep)
        chunks = []
        current_chunk = []
        current_len = 0
        
        for part in parts:
            part_len = len(part)
            # Add separator length if not first part
            add_len = part_len + (len(sep) if current_chunk else 0)
            
            if current_len + add_len <= max_chunk_size:
                current_chunk.append(part)
                current_len += add_len
            else:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                    
                # Handle single large parts
                if part_len > max_chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current_chunk = []
                    current_len = 0
                else:
                    # Implement overlap by taking the last few items of previous chunk if possible
                    # For simplicity, we just overlap by character count
                    current_chunk = [part]
                    current_len = part_len
                    
        if current_chunk:
            chunks.append(sep.join(current_chunk))
            
        return chunks

    return _split(text, separators)
