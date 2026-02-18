import re

class GRPromptReplacer:
    """
    A ComfyUI node that replaces words/phrases in text based on multi-line replacement rules.
    Each line should contain: phrase_to_replace,replacement_phrase
    Supports multi-word phrases and highlights the changes made.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter text to process..."
                }),
                "replacement_rules": ("STRING", {
                    "multiline": True,
                    "default": "blonde hair,black hair\nfair skin,tan skin\nblue eyes,brown eyes",
                    "placeholder": "Enter replacement rules (one per line):\nold phrase,new phrase\nold phrase,new phrase"
                }),
            },
            "optional": {
                "case_sensitive": ("BOOLEAN", {
                    "default": False,
                    "label": "Case Sensitive Matching"
                }),
                "match_whole_words": ("BOOLEAN", {
                    "default": True,
                    "label": "Match Whole Words Only",
                    "tooltip": "When enabled, 'man' won't match 'woman'"
                }),
                "sort_by_length": ("BOOLEAN", {
                    "default": True,
                    "label": "Sort Rules by Length (Longest First)",
                    "tooltip": "Prevents shorter phrases from replacing parts of longer phrases"
                }),
                "preserve_case": ("BOOLEAN", {
                    "default": True,
                    "label": "Preserve Original Case Pattern",
                    "tooltip": "Attempt to maintain the case pattern of the original text"
                }),
                "highlight_format": (["markdown", "html", "plain"], {
                    "default": "markdown",
                    "label": "Highlight Format"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "highlighted_text")
    FUNCTION = "replace_words"
    CATEGORY = "GR Utilities"

    def parse_rules(self, rules_text):
        """Parse the multi-line replacement rules into a list of (old, new) tuples."""
        rules = []
        lines = rules_text.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Split by comma and trim whitespace
            parts = [part.strip() for part in line.split(',', 1)]
            
            if len(parts) == 2:
                old_phrase, new_phrase = parts
                if old_phrase:  # Skip if old phrase is empty
                    # Normalize whitespace in phrases
                    old_phrase = ' '.join(old_phrase.split())
                    new_phrase = ' '.join(new_phrase.split())
                    rules.append((old_phrase, new_phrase))
            else:
                print(f"Warning: Line {line_num} in replacement rules is malformed: '{line}'")
        
        return rules

    def escape_for_regex(self, text):
        """Escape text for regex but preserve word boundaries for phrases."""
        # Escape special regex characters
        escaped = re.escape(text)
        # Replace escaped spaces with \s+ to handle multiple spaces
        escaped = escaped.replace(r'\ ', r'\s+')
        return escaped

    def create_whole_word_pattern(self, phrase):
        """Create a pattern that matches whole words, handling multi-word phrases."""
        words = phrase.split()
        if len(words) == 1:
            # Single word - use word boundaries
            return r'\b' + re.escape(words[0]) + r'\b'
        else:
            # Multi-word phrase - need to handle boundaries for first and last word
            pattern_parts = []
            for i, word in enumerate(words):
                escaped = re.escape(word)
                if i == 0:
                    # First word - check boundary before
                    pattern_parts.append(r'\b' + escaped)
                elif i == len(words) - 1:
                    # Last word - check boundary after
                    pattern_parts.append(escaped + r'\b')
                else:
                    # Middle words - no boundaries needed
                    pattern_parts.append(escaped)
            
            # Join with whitespace pattern
            return r'\s+'.join(pattern_parts)

    def preserve_case_pattern(self, matched_text, replacement):
        """Attempt to preserve the case pattern of the matched text."""
        # If replacement is empty or match is empty, return as is
        if not matched_text or not replacement:
            return replacement
        
        # Check case patterns
        if matched_text.isupper():
            # ALL CAPS
            return replacement.upper()
        elif matched_text.islower():
            # all lowercase
            return replacement.lower()
        elif matched_text[0].isupper() and not matched_text[1:].isupper():
            # Title Case or First letter capitalized
            if len(replacement) > 1:
                return replacement[0].upper() + replacement[1:].lower()
            else:
                return replacement.upper()
        elif matched_text[0].isupper() and matched_text[1:].isupper():
            # First letter capital, rest caps (unusual)
            return replacement.upper()
        else:
            # Mixed case - try to preserve pattern
            result = []
            replacement_chars = list(replacement)
            replacement_index = 0
            
            for char in matched_text:
                if replacement_index >= len(replacement_chars):
                    break
                if char.isupper():
                    result.append(replacement_chars[replacement_index].upper())
                else:
                    result.append(replacement_chars[replacement_index].lower())
                replacement_index += 1
            
            # Add any remaining replacement characters
            if replacement_index < len(replacement_chars):
                result.extend(replacement_chars[replacement_index:])
            
            return ''.join(result)

    def apply_highlighting(self, text, changes, format_type):
        """Apply highlighting to the text based on the changes made."""
        if not changes:
            return text
        
        highlighted = text
        
        if format_type == "markdown":
            # Markdown highlighting (bold)
            for old_text, new_text, start_pos, end_pos in reversed(changes):
                # We need to find the new text in the result
                # This is simplified - in practice we track positions during replacement
                highlighted = highlighted.replace(new_text, f"**{new_text}**")
        
        elif format_type == "html":
            # HTML highlighting with yellow background
            for old_text, new_text, start_pos, end_pos in reversed(changes):
                highlighted = highlighted.replace(
                    new_text, 
                    f'<span style="background-color: #ffff00; font-weight: bold;">{new_text}</span>'
                )
        
        elif format_type == "plain":
            # Plain text with brackets
            for old_text, new_text, start_pos, end_pos in reversed(changes):
                highlighted = highlighted.replace(new_text, f"[{new_text}]")
        
        return highlighted

    def replace_words_batch(self, text, rules, case_sensitive, match_whole_words, preserve_case):
        """
        Batch replacement using a single regex for all rules.
        Returns both the replaced text and a list of changes made.
        """
        if not rules:
            return text, []
        
        # Sort rules by length (longest first) to prevent partial matches
        rules.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Build pattern and replacement mapping
        pattern_parts = []
        replacement_map = {}
        
        for old_phrase, new_phrase in rules:
            # Create the pattern based on matching options
            if match_whole_words:
                pattern = self.create_whole_word_pattern(old_phrase)
            else:
                # Simple pattern that can match anywhere
                pattern = self.escape_for_regex(old_phrase)
            
            pattern_parts.append(f'({pattern})')
            
            # Store replacement with case handling
            key = old_phrase.lower() if not case_sensitive else old_phrase
            replacement_map[key] = new_phrase
        
        # Create a single pattern that matches any of the phrases
        combined_pattern = '|'.join(pattern_parts)
        
        if not case_sensitive:
            flags = re.IGNORECASE
        else:
            flags = 0
        
        changes = []
        result_parts = []
        last_end = 0
        
        # Custom replacement function that tracks changes
        def replace_func(match):
            nonlocal last_end
            
            # Add text before the match
            result_parts.append(text[last_end:match.start()])
            
            # Find which group matched and process it
            for i, group in enumerate(match.groups()):
                if group is not None:
                    matched_text = group
                    
                    # Find the matching rule
                    for old_phrase, new_phrase in rules:
                        # Check if this phrase matches
                        match_found = False
                        
                        if not case_sensitive:
                            if matched_text.lower() == old_phrase.lower():
                                match_found = True
                            elif re.match(self.create_whole_word_pattern(old_phrase) if match_whole_words else self.escape_for_regex(old_phrase), 
                                        matched_text, re.IGNORECASE):
                                match_found = True
                        else:
                            if matched_text == old_phrase:
                                match_found = True
                            elif re.match(self.create_whole_word_pattern(old_phrase) if match_whole_words else self.escape_for_regex(old_phrase), 
                                        matched_text):
                                match_found = True
                        
                        if match_found:
                            # Apply case preservation if requested
                            if preserve_case:
                                replacement = self.preserve_case_pattern(matched_text, new_phrase)
                            else:
                                replacement = new_phrase
                            
                            # Record the change
                            changes.append({
                                'old': matched_text,
                                'new': replacement,
                                'start': match.start(),
                                'end': match.end(),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })
                            
                            result_parts.append(replacement)
                            last_end = match.end()
                            return replacement
                    
                    # No rule found (shouldn't happen), return original
                    result_parts.append(matched_text)
                    last_end = match.end()
                    return matched_text
            
            # Default return
            result_parts.append(match.group(0))
            last_end = match.end()
            return match.group(0)
        
        try:
            # Perform replacements
            re.sub(combined_pattern, replace_func, text, flags=flags)
            
            # Add remaining text
            if last_end < len(text):
                result_parts.append(text[last_end:])
            
            result = ''.join(result_parts)
            
        except Exception as e:
            print(f"Regex error: {e}")
            # Fallback to simple replacement
            result = text
            changes = []
            for old_phrase, new_phrase in rules:
                if match_whole_words:
                    pattern = self.create_whole_word_pattern(old_phrase)
                    flags = 0 if case_sensitive else re.IGNORECASE
                    
                    # Simple change tracking for fallback
                    def fallback_replacer(match):
                        matched = match.group(0)
                        if preserve_case:
                            replacement = self.preserve_case_pattern(matched, new_phrase)
                        else:
                            replacement = new_phrase
                        
                        changes.append({
                            'old': matched,
                            'new': replacement,
                            'start': match.start(),
                            'end': match.end(),
                            'rule_used': f"{old_phrase} → {new_phrase}"
                        })
                        return replacement
                    
                    result = re.sub(pattern, fallback_replacer, result, flags=flags)
                else:
                    if case_sensitive:
                        # Simple find and replace with change tracking
                        pos = 0
                        while True:
                            idx = result.find(old_phrase, pos)
                            if idx == -1:
                                break
                            
                            changes.append({
                                'old': old_phrase,
                                'new': new_phrase,
                                'start': idx,
                                'end': idx + len(old_phrase),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })
                            
                            result = result[:idx] + new_phrase + result[idx + len(old_phrase):]
                            pos = idx + len(new_phrase)
                    else:
                        pattern = re.escape(old_phrase)
                        
                        def fallback_replacer_ci(match):
                            matched = match.group(0)
                            if preserve_case:
                                replacement = self.preserve_case_pattern(matched, new_phrase)
                            else:
                                replacement = new_phrase
                            
                            changes.append({
                                'old': matched,
                                'new': replacement,
                                'start': match.start(),
                                'end': match.end(),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })
                            return replacement
                        
                        result = re.sub(pattern, fallback_replacer_ci, result, flags=re.IGNORECASE)
        
        return result, changes

    def format_changes_report(self, changes, format_type):
        """Create a human-readable report of all changes made."""
        if not changes:
            return "No changes were made."
        
        if format_type == "markdown":
            report = "### Changes Made:\n\n"
            for i, change in enumerate(changes, 1):
                report += f"{i}. **{change['old']}** → **{change['new']}**  \n"
            return report
        
        elif format_type == "html":
            report = "<h3>Changes Made:</h3>\n<ul>\n"
            for change in changes:
                report += f'  <li><b>{change["old"]}</b> → <b>{change["new"]}</b></li>\n'
            report += "</ul>"
            return report
        
        else:  # plain
            report = "Changes Made:\n"
            for i, change in enumerate(changes, 1):
                report += f"{i}. {change['old']} -> {change['new']}\n"
            return report

    def replace_words(self, text, replacement_rules, case_sensitive=False, 
                     match_whole_words=True, sort_by_length=True, preserve_case=True,
                     highlight_format="markdown"):
        """Apply replacement rules to the input text and return both plain and highlighted versions."""
        if not text or not replacement_rules:
            return (text, text)
        
        # Parse rules
        rules = self.parse_rules(replacement_rules)
        
        if not rules:
            return (text, text)
        
        # Sort by length if requested (longest first to prevent partial matches)
        if sort_by_length:
            rules.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Perform replacements with change tracking
        result, changes = self.replace_words_batch(
            text, rules, case_sensitive, match_whole_words, preserve_case
        )
        
        # Create highlighted version
        if changes:
            # Sort changes by position (descending) to avoid offset issues
            changes_sorted = sorted(changes, key=lambda x: x['start'], reverse=True)
            
            highlighted = text
            for change in changes_sorted:
                old_text = change['old']
                new_text = change['new']
                
                # Apply highlighting based on format
                if highlight_format == "markdown":
                    highlighted = highlighted[:change['start']] + \
                                 f"**{new_text}**" + \
                                 highlighted[change['end']:]
                elif highlight_format == "html":
                    highlighted = highlighted[:change['start']] + \
                                 f'<span style="background-color: #ffff00; font-weight: bold;">{new_text}</span>' + \
                                 highlighted[change['end']:]
                else:  # plain
                    highlighted = highlighted[:change['start']] + \
                                 f"[{new_text}]" + \
                                 highlighted[change['end']:]
            
            # Also create a changes report
            changes_report = self.format_changes_report(changes, highlight_format)
            
            # For the second output, we can either return just the highlighted text
            # or combine it with the changes report
            if highlight_format == "markdown":
                highlighted = f"{highlighted}\n\n{changes_report}"
            elif highlight_format == "html":
                highlighted = f"{highlighted}<br><br>{changes_report}"
            else:
                highlighted = f"{highlighted}\n\n{changes_report}"
        else:
            highlighted = text
        
        return (result, highlighted)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GRPromptReplacer": GRPromptReplacer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPromptReplacer": "GR Prompt Replacer"
}