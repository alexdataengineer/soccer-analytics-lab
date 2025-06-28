# Match Insights: Soccer Analytics Lab

Welcome! This project is a fusion of my passion for football and technology. Here, I explore how computer vision and data science can reveal new insights from a full soccer match video. The goal is to transform raw match footage into meaningful statistics, tactical visualizations, and actionable insights for fans, analysts, and anyone who loves the game.

## The Business Problem

### What Traditional Soccer Analysis Lacks
- **Manual Analysis is Slow**: Coaches spend hours reviewing match footage manually
- **Limited Data Points**: Traditional stats (goals, assists) don't tell the full story
- **Subjective Evaluations**: Player performance often relies on "eye test" rather than data
- **Missed Opportunities**: Key tactical patterns go unnoticed without systematic analysis
- **High Costs**: Professional analysis tools are expensive and require specialized training

### What This Project Solves
- **Automated Insights**: Transform any match video into comprehensive analytics in minutes
- **Objective Metrics**: Quantify player movement, team tactics, and field control
- **Tactical Intelligence**: Identify patterns that human observers might miss
- **Cost-Effective**: Democratize professional-level analysis for smaller clubs and organizations
- **Scalable**: Process multiple matches simultaneously without additional resources

## Business Applications

### For Football Clubs
- **Scouting**: Evaluate players based on movement patterns and tactical intelligence
- **Performance Analysis**: Identify strengths and weaknesses in team tactics
- **Opponent Research**: Understand rival team strategies and prepare counter-tactics
- **Training Optimization**: Design practice sessions based on actual match data
- **Recruitment**: Make data-driven decisions about player acquisitions

### For Broadcasters & Media
- **Enhanced Commentary**: Provide real-time statistics and insights during broadcasts
- **Post-Match Analysis**: Create engaging content with detailed tactical breakdowns
- **Fan Engagement**: Offer interactive statistics and visualizations to viewers
- **Content Creation**: Generate social media content with unique match insights

### For Sports Organizations
- **Youth Development**: Track player development with objective metrics
- **Competition Analysis**: Compare teams and players across different leagues
- **Revenue Generation**: Offer premium analytics services to clubs and media
- **Performance Tracking**: Monitor long-term trends and improvements

### For Betting & Fantasy Sports
- **Performance Prediction**: Use historical data to predict player and team performance
- **Risk Assessment**: Evaluate player fitness and form through movement analysis
- **Market Intelligence**: Identify undervalued players based on tactical contributions

## Project Structure

```
soccer-analytics-lab/
├── src/                    # Core modules
│   ├── detection.py        # YOLOv8 object detection
│   ├── tracking.py         # Player and ball tracking
│   ├── analysis.py         # Data analysis and statistics
│   ├── visualization.py    # Charts and visualizations
│   ├── dashboard.py        # Streamlit web dashboard
│   ├── player_clustering.py # Player style clustering (KMeans)
│   ├── pass_detection.py   # Automatic pass recognition
│   ├── epv_model.py        # Expected Possession Value
│   └── attack_pressure.py  # Attack and pressure detection
├── notebooks/              # Jupyter notebooks for exploration
│   └── 01_exploration.ipynb
├── data/                   # Input video files and raw data
├── outputs/                # Generated results and visualizations
│   ├── tracking_results.json
│   ├── analysis_results.json
│   ├── player_clusters.png
│   ├── epv_heatmap.png
│   └── match_momentum.png
├── venv/                   # Python virtual environment
├── requirements.txt        # Project dependencies
├── run_analysis.py         # Main analysis pipeline
├── setup.py               # Project configuration
├── test_setup.py          # Setup verification tests
├── yolov8n.pt             # YOLOv8 model weights
└── README.md              # Project documentation
```
<img width="301" alt="image" src="https://github.com/user-attachments/assets/1a70b9b3-0a85-4a41-9425-34fb7ff5c940" />

## Why I Built This

As a football fan and tech enthusiast, I always wondered: what if I could break down a match like a professional analyst? I wanted to see not just who scored, but how teams built their plays, who controlled the field, and which players had unique styles. This project is my way of answering those questions using modern AI tools.

## What It Does

- **Player and Ball Detection:** Uses YOLOv8 and OpenCV to detect and track all players and the ball throughout the match.
- **Player Style Clustering:** Groups players by their movement and activity profiles (e.g., box-to-box, defensive, explosive) using KMeans clustering.
- **Automatic Pass Recognition:** Identifies passes by tracking the ball and player proximity, logging who passed to whom, when, and how far.
- **Expected Possession Value (EPV):** Estimates the value of each field position, showing which areas are most likely to lead to goals.
- **Attack and Pressure Detection:** Detects moments when a team pushes forward with multiple players or applies collective pressure.
- **Key Play Replays:** (Planned) Automatically saves video clips of key plays, like fast advances or passing sequences.
- **Interactive Dashboard:** All insights are visualized in a Streamlit dashboard, with charts, heatmaps, and timelines.

## Key Results (First 15 Minutes Example)

- 46 players tracked, 449 ball positions detected
- 100% possession for Manchester City in the first 15 minutes
- Player clusters revealed distinct tactical roles
- Pass maps and EPV heatmaps highlighted team strategies and danger zones
- Attack and pressure moments visualized as momentum swings

  ![image](https://github.com/user-attachments/assets/17313adf-6f71-40ec-a3a6-e1f6a083a980)
  ![image](https://github.com/user-attachments/assets/f7f64513-98a3-4087-8f85-6d3db251e044)
  ![image](https://github.com/user-attachments/assets/3ea3e4c2-fb73-40a5-a5da-661da4e5d0ff)
  ![image](https://github.com/user-attachments/assets/3ace228e-6bd2-4b93-8f10-0f3c1f321c4b)
  ![image](https://github.com/user-attachments/assets/a50b0502-8aeb-4659-8006-9fb90499b464)



## ROI & Business Value

### Cost Savings
- **90% reduction** in manual analysis time
- **Automated processing** of multiple matches simultaneously
- **No specialized training** required to use the system

### Performance Improvements
- **Data-driven decisions** replace subjective evaluations
- **Tactical optimization** through pattern recognition
- **Competitive advantage** through advanced analytics

### Revenue Opportunities
- **SaaS Platform**: Subscription-based analytics service
- **Consulting Services**: Expert analysis and insights
- **Data Licensing**: Sell anonymized match statistics
- **Integration Services**: Custom implementations for clubs

## What I Learned

- How to combine computer vision, tracking, and clustering to extract tactical insights
- The challenges of real-world video (occlusion, detection errors, etc.)
- The value of visual analytics for understanding the flow of a match
- The importance of translating technical capabilities into business value

## Technology Stack

- Python, OpenCV, Ultralytics YOLOv8
- Scikit-learn, Pandas, Matplotlib, Seaborn
- Streamlit for the dashboard

## Next Steps

- Improve player identification and team assignment
- Add automatic video replays for key plays
- Refine tactical event detection (e.g., pressing, counter-attacks)
- Develop API for integration with existing club systems
- Scale for real-time processing during live matches

## About Me

I'm a football lover and tech explorer, always looking for new ways to connect data and the beautiful game. This project is a personal journey into sports analytics, and I hope it inspires others to see football through a new lens.

---

If you have questions or want to discuss football analytics, feel free to reach out! 
