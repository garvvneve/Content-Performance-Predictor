from flask import Flask, render_template_string, request
import joblib
import numpy as np

model = joblib.load("model/content_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graphura AI | Content Intelligence</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            color: #334155;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .app-container {
            display: flex;
            gap: 30px;
            max-width: 1100px;
            width: 100%;
        }

        .form-container {
            flex: 1;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            border: 1px solid #e2e8f0;
        }

        .insights-container {
            flex: 1;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            border: 1px solid #e2e8f0;
        }

        .app-header {
            text-align: center;
            margin-bottom: 40px;
        }

        .app-header h1 {
            font-size: 28px;
            font-weight: 700;
            color: #475569;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .app-header p {
            color: #64748b;
            font-size: 15px;
            font-weight: 400;
        }

        .form-section {
            margin-bottom: 30px;
        }

        .form-section h3 {
            font-size: 14px;
            font-weight: 600;
            color: #64748b;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .form-group {
            margin-bottom: 24px;
        }

        .form-label {
            display: block;
            font-size: 15px;
            font-weight: 500;
            color: #475569;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .form-label-icon {
            width: 24px;
            height: 24px;
            background: #f59e0b;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
        }

        select, input {
            width: 100%;
            padding: 14px 16px;
            background: #f8fafc;
            color: #334155;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 15px;
            font-family: inherit;
            transition: all 0.2s ease;
        }

        select:focus, input:focus {
            outline: none;
            border-color: #f59e0b;
            background: white;
            box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.1);
        }

        select {
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 16px center;
            background-size: 16px;
            padding-right: 40px;
        }

        .input-with-example {
            position: relative;
        }

        .example-text {
            position: absolute;
            right: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: #94a3b8;
            font-size: 14px;
            pointer-events: none;
        }

        .example-section {
            background: #fff7ed;
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
            border-left: 4px solid #f59e0b;
        }

        .example-section h4 {
            color: #d97706;
            font-size: 16px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .example-section p {
            color: #57534e;
            font-size: 14px;
            line-height: 1.6;
        }

        .submit-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }

        .submit-btn:hover {
            background: linear-gradient(135deg, #d97706, #b45309);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(217, 119, 6, 0.2);
        }

        .result-container {
            margin-top: 30px;
            animation: slideUp 0.3s ease;
        }

        .result-card {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }

        .result-score {
            font-size: 56px;
            font-weight: 800;
            margin-bottom: 10px;
            line-height: 1;
        }

        .result-high .result-score {
            background: linear-gradient(135deg, #10b981, #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .result-medium .result-score {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .result-low .result-score {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .result-label {
            font-size: 24px;
            font-weight: 700;
            color: #334155;
            margin-bottom: 12px;
        }

        .result-description {
            color: #64748b;
            font-size: 15px;
            line-height: 1.6;
            font-weight: 500;
        }

        .insights-header {
            margin-bottom: 30px;
        }

        .insights-header h2 {
            font-size: 24px;
            font-weight: 700;
            color: #475569;
            margin-bottom: 8px;
        }

        .insights-header p {
            color: #64748b;
            font-size: 15px;
        }

        .tip-card {
            background: #f8fafc;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid transparent;
            transition: transform 0.2s ease;
        }

        .tip-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
        }

        .tip-card.platform {
            border-left-color: #10b981;
        }

        .tip-card.timing {
            border-left-color: #f59e0b;
        }

        .tip-card.format {
            border-left-color: #8b5cf6;
        }

        .tip-card.hashtags {
            border-left-color: #ef4444;
        }

        .tip-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .tip-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .tip-card.platform .tip-icon {
            background: #d1fae5;
            color: #10b981;
        }

        .tip-card.timing .tip-icon {
            background: #fef3c7;
            color: #d97706;
        }

        .tip-card.format .tip-icon {
            background: #ede9fe;
            color: #8b5cf6;
        }

        .tip-card.hashtags .tip-icon {
            background: #fee2e2;
            color: #dc2626;
        }

        .tip-title {
            font-size: 16px;
            font-weight: 600;
            color: #334155;
        }

        .tip-content {
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
        }

        .system-info {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
        }

        .system-title {
            font-size: 14px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .system-desc {
            color: #94a3b8;
            font-size: 13px;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 900px) {
            .app-container {
                flex-direction: column;
            }
            
            .form-container,
            .insights-container {
                padding: 30px;
            }
        }
    </style>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="app-container">
        <!-- Left Panel: Prediction Form -->
        <div class="form-container">
            <div class="app-header">
                <h1>Content Performance Predictor</h1>
                <p>AI-powered insights for optimal content strategy</p>
            </div>

            <form method="post">
                <!-- Platform Selection -->
                <div class="form-section">
                    <h3>Platform Configuration</h3>
                    <div class="form-group">
                        <label class="form-label">
                            <span class="form-label-icon">📱</span>
                            Platform
                        </label>
                        <div class="input-with-example">
                            <select name="platform_code" class="platform-select">
                                <option value="0" {% if platform == 0 %}selected{% endif %}>Instagram</option>
                                <option value="1" {% if platform == 1 %}selected{% endif %}>LinkedIn</option>
                            </select>
                            <div class="example-text">Select social platform</div>
                        </div>
                    </div>

                    <!-- Example -->
                    <div class="example-section">
                        <h4>💡 Example</h4>
                        <p id="platform-example">
                            {% if platform == 0 %}
                                For maximum engagement: <strong>Instagram</strong> is best for visual content, stories, and reels. Use vertical videos for better performance.
                            {% elif platform == 1 %}
                                For maximum engagement: <strong>LinkedIn</strong> is ideal for professional content, articles, and industry insights. Keep it formal and informative.
                            {% else %}
                                For maximum engagement: Choose <strong>Instagram</strong> for visual content, <strong>LinkedIn</strong> for professional insights.
                            {% endif %}
                        </p>
                    </div>
                </div>

                <!-- Content Type -->
                <div class="form-section">
                    <h3>Content Details</h3>
                    <div class="form-group">
                        <label class="form-label">
                            <span class="form-label-icon">🎬</span>
                            Post Type
                        </label>
                        <div class="input-with-example">
                            <select name="post_type_code" class="post-type-select">
                                <option value="0" {% if post_type == 0 %}selected{% endif %}>Image</option>
                                <option value="1" {% if post_type == 1 %}selected{% endif %}>Carousel</option>
                                <option value="2" {% if post_type == 2 %}selected{% endif %}>Video</option>
                                <option value="3" {% if post_type == 3 %}selected{% endif %}>Text</option>
                            </select>
                            <div class="example-text">Select content format</div>
                        </div>
                    </div>

                    <!-- Example -->
                    <div class="example-section">
                        <h4>📊 Example</h4>
                        <p id="content-example">
                            {% if post_type == 2 %}
                                For higher engagement: <strong>Video content</strong> gets 2.3x more views than images. Keep videos under 60 seconds for maximum retention.
                            {% elif post_type == 1 %}
                                For higher engagement: <strong>Carousel posts</strong> are great for tutorials and step-by-step guides. Each slide tells part of the story.
                            {% else %}
                                For higher engagement: Use <strong>Video</strong> (gets 2.3x more views) or <strong>Carousel</strong> for educational content.
                            {% endif %}
                        </p>
                    </div>
                </div>

                <!-- Timing -->
                <div class="form-section">
                    <h3>Timing Strategy</h3>
                    <div class="form-group">
                        <label class="form-label">
                            <span class="form-label-icon">📅</span>
                            Day of Week
                        </label>
                        <div class="input-with-example">
                            <select name="day_code" class="day-select">
                                <option value="0" {% if day == 0 %}selected{% endif %}>Monday</option>
                                <option value="1" {% if day == 1 %}selected{% endif %}>Tuesday</option>
                                <option value="2" {% if day == 2 %}selected{% endif %}>Wednesday</option>
                                <option value="3" {% if day == 3 %}selected{% endif %}>Thursday</option>
                                <option value="4" {% if day == 4 %}selected{% endif %}>Friday</option>
                                <option value="5" {% if day == 5 %}selected{% endif %}>Saturday</option>
                                <option value="6" {% if day == 6 %}selected{% endif %}>Sunday</option>
                            </select>
                            <div class="example-text">When to post</div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="form-label">
                            <span class="form-label-icon">🏷️</span>
                            Hashtag Count
                        </label>
                        <div class="input-with-example">
                            <input type="number" 
                                   name="hashtag_count" 
                                   min="0" 
                                   max="30" 
                                   value="{% if hashtag_count %}{{ hashtag_count }}{% else %}5{% endif %}" 
                                   required
                                   class="hashtag-input">
                            <div class="example-text" id="hashtag-example">
                                {% if hashtag_count %}
                                    {% if platform == 0 %}
                                        {% if hashtag_count >= 5 and hashtag_count <= 10 %}
                                            ✓ Ideal for Instagram
                                        {% elif hashtag_count > 10 %}
                                            ⚠️ Too many for Instagram
                                        {% else %}
                                            ⚠️ Too few for Instagram
                                        {% endif %}
                                    {% elif platform == 1 %}
                                        {% if hashtag_count >= 3 and hashtag_count <= 5 %}
                                            ✓ Perfect for LinkedIn
                                        {% elif hashtag_count > 5 %}
                                            ⚠️ Too many for LinkedIn
                                        {% else %}
                                            ⚠️ Too few for LinkedIn
                                        {% endif %}
                                    {% else %}
                                        0-30 hashtags
                                    {% endif %}
                                {% else %}
                                    0-30 hashtags
                                {% endif %}
                            </div>
                        </div>
                    </div>

                    <!-- Example -->
                    <div class="example-section">
                        <h4>⏰ Example</h4>
                        <p id="timing-example">
                            {% if platform == 0 %}
                                {% if day == 5 or day == 6 %}
                                    Best timing: <strong>Weekends</strong> are perfect for Instagram. Post between 9 AM - 2 PM for maximum visibility. Use <strong>7-10 hashtags</strong> for optimal reach.
                                {% else %}
                                    Best timing: <strong>Weekends (Sat-Sun)</strong> perform best on Instagram. Consider adjusting your posting schedule.
                                {% endif %}
                            {% elif platform == 1 %}
                                {% if day >= 1 and day <= 3 %}
                                    Best timing: <strong>Tuesday-Thursday</strong> are ideal for LinkedIn. Post during business hours (8 AM - 12 PM). Use <strong>3-5 professional hashtags</strong>.
                                {% else %}
                                    Best timing: <strong>Tuesday-Thursday</strong> perform best on LinkedIn. Consider posting mid-week.
                                {% endif %}
                            {% else %}
                                Best timing: <strong>Weekends</strong> for Instagram, <strong>Tuesday-Thursday</strong> for LinkedIn. Use <strong>5-10 hashtags</strong> on Instagram, <strong>3-5</strong> on LinkedIn.
                            {% endif %}
                        </p>
                    </div>
                </div>

                <!-- Submit Button -->
                <button type="submit" class="submit-btn">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"/>
                    </svg>
                    Analyze Performance
                </button>

                <!-- Results - Shows REAL prediction -->
                {% if result and probability %}
                <div class="result-container">
                    <div class="result-card {% if css == 'high' %}result-high{% elif css == 'medium' %}result-medium{% else %}result-low{% endif %}">
                        <div class="result-score">
                            {{ "%.1f"|format(probability) }}%
                        </div>
                        <div class="result-label">{{ result }}</div>
                        <div class="result-description">
                            {% if css == 'high' %}
                                🚀 Excellent content strategy! 
                                {% if platform == 0 and day in [5,6] %}
                                    Weekend posting on Instagram is a great choice!
                                {% elif platform == 1 and day in [1,2,3] %}
                                    Mid-week posting on LinkedIn is optimal.
                                {% endif %}
                            {% elif css == 'medium' %}
                                📊 Good potential with room for optimization.
                                {% if platform == 0 and hashtag_count < 5 %}
                                    Try increasing hashtags to 5-10 for better reach.
                                {% elif platform == 1 and hashtag_count > 5 %}
                                    LinkedIn performs better with 3-5 hashtags.
                                {% endif %}
                            {% else %}
                                💡 Consider adjusting your strategy.
                                {% if platform == 0 and day not in [5,6] %}
                                    Instagram works best on weekends!
                                {% elif platform == 1 and day in [0,1] %}
                                    Avoid Monday/Tuesday for LinkedIn posts.
                                {% endif %}
                            {% endif %}
                        </div>
                    </div>
                </div>
                {% endif %}
            </form>
        </div>

        <!-- Right Panel: Insights -->
        <div class="insights-container">
            <div class="insights-header">
                <h2>Content Optimization Guide</h2>
                <p>Professional insights based on 10,000+ successful posts</p>
            </div>

            <!-- Platform Tips -->
            <div class="tip-card platform">
                <div class="tip-header">
                    <div class="tip-icon">📱</div>
                    <div class="tip-title">Platform Strategy</div>
                </div>
                <div class="tip-content">
                    <strong>Instagram:</strong> Visual-first platform. Best for B2C, lifestyle, and creative content.<br>
                    <strong>LinkedIn:</strong> Professional network. Ideal for B2B, career insights, and industry news.
                </div>
            </div>

            <!-- Timing Tips -->
            <div class="tip-card timing">
                <div class="tip-header">
                    <div class="tip-icon">⏰</div>
                    <div class="tip-title">Optimal Posting Times</div>
                </div>
                <div class="tip-content">
                    • <strong>Instagram:</strong> Saturday-Sunday, 9 AM - 2 PM<br>
                    • <strong>LinkedIn:</strong> Tuesday-Thursday, 8 AM - 12 PM<br>
                    • Avoid Monday mornings across all platforms
                </div>
            </div>

            <!-- Format Tips -->
            <div class="tip-card format">
                <div class="tip-header">
                    <div class="tip-icon">🎬</div>
                    <div class="tip-title">Content Format Best Practices</div>
                </div>
                <div class="tip-content">
                    • <strong>Video:</strong> Highest engagement (2.3x more than images)<br>
                    • <strong>Carousel:</strong> Great for tutorials and step-by-step guides<br>
                    • <strong>Images:</strong> Use high-quality, relevant visuals<br>
                    • <strong>Text:</strong> Best for thought leadership on LinkedIn
                </div>
            </div>

            <!-- Hashtag Tips -->
            <div class="tip-card hashtags">
                <div class="tip-header">
                    <div class="tip-icon">🏷️</div>
                    <div class="tip-title">Hashtag Strategy</div>
                </div>
                <div class="tip-content">
                    • <strong>Instagram:</strong> 5-10 relevant hashtags (mix of popular and niche)<br>
                    • <strong>LinkedIn:</strong> 3-5 industry-specific hashtags<br>
                    • Use location-based hashtags for local businesses<br>
                    • Research trending hashtags weekly
                </div>
            </div>

            <!-- System Info -->
            <div class="system-info">
                <div class="system-title">Powered by Graphura AI</div>
                <div class="system-desc">Random Forest ML Model • Real-time Analysis • Enterprise Grade</div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.querySelector('form');
            const button = form.querySelector('.submit-btn');
            
            // Form submission animation
            form.addEventListener('submit', function(e) {
                const originalHTML = button.innerHTML;
                button.innerHTML = `
                    <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10" stroke-opacity="0.3"/>
                        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"/>
                    </svg>
                    Processing Analysis...
                `;
                button.disabled = true;
                
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                    .spinner {
                        animation: spin 1s linear infinite;
                    }
                `;
                document.head.appendChild(style);
            });
        });
    </script>
</body>
</html>
"""

def apply_business_rules(prob, platform, day, hashtags):
    adjustment = 0

    # LinkedIn rules
    if platform == 1:
        if hashtags > 5:
            adjustment -= 0.10
        if day in [0,1]:  # Monday, Tuesday
            adjustment -= 0.05

    # Instagram rules
    if platform == 0:
        if day in [5,6]:  # Weekend
            adjustment += 0.05
        if hashtags < 3:
            adjustment -= 0.05

    final_prob = max(0.05, min(prob + adjustment, 0.95))
    return final_prob

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    css = None
    probability = None
    platform = None
    post_type = None
    day = None
    hashtag_count = None

    if request.method == "POST":
        platform = int(request.form["platform_code"])
        post_type = int(request.form["post_type_code"])
        day = int(request.form["day_code"])
        hashtag_count = int(request.form["hashtag_count"])

        data = {
            "platform_code": platform,
            "post_type_code": post_type,
            "day_code": day,
            "hashtag_count": hashtag_count
        }

        row = [data.get(f, 0) for f in feature_names]
        X = scaler.transform([row])

        base_prob = model.predict_proba(X)[0][1]
        final_prob = apply_business_rules(base_prob, platform, day, hashtag_count)
        
        probability = final_prob * 100

        if final_prob >= 0.70:
            result = f"🔥 High Potential"
            css = "high"
        elif final_prob >= 0.45:
            result = f"⚠️ Medium Potential"
            css = "medium"
        else:
            result = f"❌ Low Potential"
            css = "low"

    return render_template_string(
        HTML, 
        result=result, 
        css=css,
        probability=probability,
        platform=platform,
        post_type=post_type,
        day=day,
        hashtag_count=hashtag_count
    )

if __name__ == "__main__":
    print("="*50)
    print("🚀 Graphura AI Content Predictor")
    print("="*50)
    print("✅ Model loaded successfully!")
    print("🌐 Running on http://localhost:5000")
    print("="*50)
    app.run(debug=True)