# FreeBall Project

## Project Overview
**Purpose**: To create a volleyball AI assistant, enabling their improvements through statistics, replays, and clipping. 
**Data**: Youtube video uploads from my channel

### Tech Stack
Frontend: React, FastAPI
Backend: Python, Pytorch, OpenCV
Data: --To be decided--


## Key AI/ML Functionality
1. Rally Detection: Use contextual cues from the video to determine the start/end of a rally. This may include player positions, whistles (during competition or when casual games have whistles) or ball detection

2. Player Detection: Being able to identify all players on courts, their positions (open, middle, setter, libero)

3. Action Detection: Identification of the action a player may be currently taking (spiking, receiving, blocking) and the result of said action (spike kill, receive quality, block kill)


## App Features
1. Rally Breakdown
- Breaks down a volleyball games into its individual rallies. Each rally is displayed as a small card. When clicked on, the video playback jumps to the start time of the requested rally.

2. Statistic Tracking
- Keep track of an individual player's statistics, such as number of spikes broken down by each result (spike kill, attack picked up, spike blocked)
- Able to jump to the next instance in the video of a player performing an action
- Can be scoped down to focus on the team in frame since its likely that only one team will be fully visible in the video frame


## Project Backbone
### Prototype
- Each AI/ML functionality should be working. Scope may be limited, but plans for extension of scope is in place.
- Proposed metrics for performance monitoring passes the prototype benchmark

### Proof of Concept
- Individual App features are woven together through each AI/ML functionality.
- Frontend is setup for users to interact with. Use of mockups is allowed where needed
- Latency/performance may be de-prioritised

### MVP
- App Features are working to a reasonably well degree. Metrics include performance accuracy, latency, compute reliance
- Optimisations should be considered


## Project Directory
project root/
├── .github/                       # CI/CD workflows, automation scripts
├── literature/                    # Literature Review
├── docs/                          # Project documentation for humans & AI
│   ├── architecture/              # System diagrams & data flows
│   ├── prd/                       # Product Requirements Document / specs
│   └── decisions/                 # Architecture Decision Records (ADRs)
│
├── frontend/                      # User interface (React)
│   ├── public/                    # Static assets (images, icons)
│   ├── src/
│   │   ├── app/                   # Routes and views
│   │   ├── components/            # Reusable UI components
│   │   ├── hooks/                 # Custom React hooks
│   │   ├── lib/                   # API client & fetch wrappers
│   │   └── types/                 # TypeScript interfaces
│   ├── package.json
│   └── tailwind.config.js
│
├── backend/                       # API server & business logic (FastAPI)
│   ├── src/
│   │   ├── controllers/           # Request handlers
│   │   ├── functionality/
│   │   │   ├── rally-detection/
│   │   │   ├── player-detection/
│   │   │   ├── action-detection/
│   │   │   └── utils/
│   │   ├── routes/                # API endpoints
│   │   ├── services/              # Core business logic & third-party integrations
│   │   ├── middlewares/           # Auth, rate-limiting, error handling
│   │   └── utils/                 # Helper functions
│   ├── Dockerfile
│   └── pyproject.toml
│
├── data/                          # Data engineering, vector stores, and datasets 
│   ├── raw/                       # Unprocessed source files (CSV, JSON, PDFs)
│   ├── processed/                 # Cleaned or transformed datasets
│   ├── schemas/                   # Database / Prisma / Pydantic schemas
│   └── migrations/                # Database migration scripts
│
├── docker-compose.yml             # Local multi-service orchestration
├── .CLAUDE.md                     # AI context router & coding rules
└── README.md