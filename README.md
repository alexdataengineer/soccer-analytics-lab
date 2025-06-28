# Match Insights: Soccer Analytics Lab

Welcome! This project is a fusion of my passion for football and technology. Here, I explore how computer vision and data science can reveal new insights from a full soccer match video. The goal is to transform raw match footage into meaningful statistics, tactical visualizations, and actionable insights for fans, analysts, and anyone who loves the game.

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

## What I Learned

- How to combine computer vision, tracking, and clustering to extract tactical insights
- The challenges of real-world video (occlusion, detection errors, etc.)
- The value of visual analytics for understanding the flow of a match

## Technology Stack

- Python, OpenCV, Ultralytics YOLOv8
- Scikit-learn, Pandas, Matplotlib, Seaborn
- Streamlit for the dashboard

## Next Steps

- Improve player identification and team assignment
- Add automatic video replays for key plays
- Refine tactical event detection (e.g., pressing, counter-attacks)

## About Me

I'm a football lover and tech explorer, always looking for new ways to connect data and the beautiful game. This project is a personal journey into sports analytics, and I hope it inspires others to see football through a new lens.

---

If you have questions or want to discuss football analytics, feel free to reach out! 