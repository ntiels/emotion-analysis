def get_word_level_shap_values(text, model, tokenizer, max_length=100):
    """Get SHAP values for individual words in the text"""
    try:
        words = text.split()
        if len(words) == 0:
            return None
            
        # Create word-level explanations by masking individual words
        word_contributions = np.zeros((len(words), len(emotion_names_list)))
        
        # Get baseline prediction (empty text)
        baseline_input = preprocess_text_app("", tokenizer, max_length)
        baseline_pred = model.predict(baseline_input, verbose=0)[0]
        
        # Get full text prediction
        full_input = preprocess_text_app(text, tokenizer, max_length)
        full_pred = model.predict(full_input, verbose=0)[0]
        
        # Calculate contribution of each word by removing it
        for i, word in enumerate(words):
            # Create text without this word
            masked_words = words[:i] + words[i+1:]
            masked_text = " ".join(masked_words)
            
            if masked_text.strip():
                masked_input = preprocess_text_app(masked_text, tokenizer, max_length)
                masked_pred = model.predict(masked_input, verbose=0)[0]
            else:
                masked_pred = baseline_pred
            
            # Contribution = full_prediction - prediction_without_word
            word_contributions[i] = full_pred - masked_pred
        
        return word_contributions
        
    except Exception as e:
        print(f"Error in word-level SHAP: {e}")
        return None

def create_colored_text_html(text, shap_values, emotion_names, predicted_emotion):
    """Create HTML with color-coded words based on SHAP values - only highlight important words"""
    words = text.split()
    
    # Color scheme for different emotions
    emotion_colors = {
        'joy': '#FFD700',      # Gold
        'love': '#FF69B4',     # Hot Pink
        'anger': '#FF4500',    # Red Orange
        'fear': '#8A2BE2',     # Blue Violet
        'sadness': '#4169E1',  # Royal Blue
        'surprise': '#32CD32', # Lime Green
        'neutral': '#808080'   # Gray
    }
    
    # Calculate importance threshold based on the data
    all_contributions = []
    for i in range(len(words)):
        if i < len(shap_values):
            word_contribs = shap_values[i]
            max_contrib = np.max(np.abs(word_contribs))
            all_contributions.append(max_contrib)
    
    if all_contributions:
        # Use a more dynamic threshold - only highlight words in top 30% of contributions
        # or those with absolute contribution > 5% of max contribution
        max_overall_contrib = max(all_contributions) if all_contributions else 0
        percentile_threshold = np.percentile(all_contributions, 70) if len(all_contributions) > 3 else 0
        min_threshold = max(0.05 * max_overall_contrib, 0.02)  # At least 5% of max or 0.02
        
        # Use the higher of the two thresholds
        importance_threshold = max(percentile_threshold, min_threshold)
    else:
        importance_threshold = 0.02
    
    html_parts = []
    highlighted_count = 0
    
    for i, word in enumerate(words):
        if i < len(shap_values):
            word_contribs = shap_values[i]
            max_contrib_idx = np.argmax(np.abs(word_contribs))
            max_emotion = emotion_names[max_contrib_idx]
            contribution = word_contribs[max_contrib_idx]
            abs_contribution = abs(contribution)
            
            # Only highlight if contribution is above the importance threshold
            if abs_contribution > importance_threshold:
                color = emotion_colors.get(max_emotion, '#808080')
                # Scale opacity more aggressively for better visual distinction
                opacity = min(0.3 + (abs_contribution / max_overall_contrib * 0.7), 1.0)
                
                html_parts.append(
                    f'<span style="background-color: {color}; '
                    f'opacity: {opacity}; padding: 2px 4px; margin: 1px; '
                    f'border-radius: 3px; font-weight: bold;" '
                    f'title="{max_emotion}: {contribution:.3f}">{word}</span>'
                )
                highlighted_count += 1
            else:
                html_parts.append(f'<span style="margin: 1px;">{word}</span>')
        else:
            html_parts.append(f'<span style="margin: 1px;">{word}</span>')
    
    # Add info about highlighting
    result_html = ' '.join(html_parts)
    if highlighted_count > 0:
        info_text = f"<br><small style='color: #666; font-style: italic;'>Highlighted {highlighted_count} most important words (threshold: {importance_threshold:.3f})</small>"
        result_html += info_text
    
    return result_html

# Modified section in the main analysis code:
# Replace the existing SHAP Analysis Section with this enhanced version:

# SHAP Analysis Section
st.markdown("### 🔍 Word-Level Analysis")
st.markdown("See which words contribute most to each emotion prediction:")

with st.spinner("🧠 Analyzing word contributions..."):
    # Get SHAP values for the input text
    shap_values = analyze_with_shap(user_input, model, tokenizer, emotion_names_list)
    
    if shap_values is not None:
        # Create color-coded text
        colored_html = create_colored_text_html(user_input, shap_values, emotion_names_list, max_emotion)
        
        # Display SHAP analysis
        st.markdown(f"""
        <div class="shap-container">
            <h4 style="margin-top: 0; color: #333;">💡 Word Contributions</h4>
            <div class="shap-text">
                {colored_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced legend with importance note
        st.markdown("""
        <div class="legend-container">
            <strong>Color Legend (Only Important Words Highlighted):</strong>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FFD700;"></div>
                <span>Joy</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FF69B4;"></div>
                <span>Love</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FF4500;"></div>
                <span>Anger</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #8A2BE2;"></div>
                <span>Fear</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #4169E1;"></div>
                <span>Sadness</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #32CD32;"></div>
                <span>Surprise</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #808080;"></div>
                <span>Neutral</span>
            </div>
        </div>
        <p style="font-size: 0.9rem; color: #666; margin-top: 10px;">
            <em>Note: Only words with significant contribution to the prediction are highlighted. 
            Words with minimal impact remain unhighlighted for clarity.</em>
        </p>
        """, unsafe_allow_html=True)
        
        # Enhanced detailed word contributions - show only important words
        st.markdown("#### 📊 Most Important Word Contributions")
        
        # Create a DataFrame for word contributions
        words = user_input.split()
        word_data = []
        
        for i, word in enumerate(words):
            if i < len(shap_values):
                for j, emotion in enumerate(emotion_names_list):
                    contrib = shap_values[i, j]
                    if abs(contrib) > 0.02:  # Only include meaningful contributions
                        word_data.append({
                            'Word': word,
                            'Emotion': emotion,
                            'Contribution': contrib,
                            'Abs_Contribution': abs(contrib)
                        })
        
        if word_data:
            contrib_df = pd.DataFrame(word_data)
            
            # Show top contributing words overall (across all emotions)
            top_words_overall = contrib_df.nlargest(8, 'Abs_Contribution')
            
            if not top_words_overall.empty:
                st.markdown("**Most influential words in the text:**")
                for _, row in top_words_overall.iterrows():
                    contribution_dir = "↑" if row['Contribution'] > 0 else "↓"
                    st.write(f"• **{row['Word']}** → {row['Emotion']} {contribution_dir} {abs(row['Contribution']):.3f}")
            
            # Show top contributing words for the predicted emotion
            top_contrib = contrib_df[contrib_df['Emotion'] == max_emotion].nlargest(5, 'Contribution')
            
            if not top_contrib.empty:
                st.markdown(f"**Words most supporting '{max_emotion}' prediction:**")
                for _, row in top_contrib.iterrows():
                    if row['Contribution'] > 0.01:
                        st.write(f"• **{row['Word']}**: +{row['Contribution']:.3f}")
        else:
            st.info("No words showed significant contribution above the threshold.")
    else:
        st.warning("⚠️ Could not perform SHAP analysis. Showing results without word-level explanations.")
