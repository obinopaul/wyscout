# Wyscout Multi-Agent AI System

**A sophisticated, multi-agent framework for advanced soccer data analysis and interaction using the Wyscout API and LangGraph.**

---

### **Project Status: Under Active Development** 🚧

**Current Phase: Foundational Tool Development**

This project is in its early stages. The current focus is on building a comprehensive, robust, and highly advanced set of structured tools using `langchain` and `pydantic`. These tools serve as the fundamental building blocks upon which the multi-agent system will be constructed. The core logic for agent interaction, state management, and final output generation is the next major phase of development.

---

## 1. Overview

The Wyscout Multi-Agent AI System is an ambitious project designed to create a powerful, conversational interface for accessing and reasoning over the vast and detailed Wyscout soccer database. Traditional API interactions are powerful but require technical expertise. This project abstracts that complexity away, allowing users (from fans to professional analysts) to ask complex, natural language questions and receive insightful, data-driven answers.

Our vision is a system of specialized AI agents, each an expert in a specific domain of soccer analysis (e.g., Player Scouting, Match Analysis, Tactical Analysis). These agents will collaborate, using the foundational tools documented below, to handle queries that would be impossible for a single monolithic agent to solve.

## 2. Core Components

* **Wyscout API**: The source of our ground-truth data, providing granular information on players, teams, matches, competitions, and more.
* **LangGraph**: The engine powering our multi-agent framework. We leverage LangGraph to define and manage complex, cyclical, and stateful interactions between our specialized agents.
* **LangChain & Pydantic**: The backbone of our **Structured Tools**. We use `pydantic` to create highly advanced, self-documenting, and robust schemas for our tools, ensuring that the agents can use them reliably and efficiently.

## 3. LangGraph Tools: The Foundation

The core of this project is a suite of meticulously designed tools. Our design philosophy emphasizes creating single, powerful, unified tools for each API resource rather than a multitude of small, single-purpose functions. This reduces agent confusion and promotes more efficient, parallelized data fetching.

Each tool is built with a robust, asynchronous core (`aiohttp`) and a highly-detailed `pydantic` input schema for validation and clarity.

### The Tool Suite

| Tool Name                    | Description                                                                                                                                                                                              | Status      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `wyscout_id_search`          | **The definitive search tool.** Finds the unique `wyId` for any entity (player, team, competition, referee) by name. Returns a list of potential matches to handle ambiguity.                                | ✅ Complete |
| `wyscout_player_info`        | A unified tool to get comprehensive info about a player, including career, contract, fixtures, and transfer history.                                                                                         | ✅ Complete |
| `wyscout_team_info`          | A unified tool to get all data for a team, including details, career, squad, fixtures, matches, and transfers.                                                                                               | ✅ Complete |
| `wyscout_season_info`        | A powerhouse tool to get all data for a season, including standings, scorers, assist leaders, teams, players, and more.                                                                                      | ✅ Complete |
| `wyscout_match_info`         | A unified tool to get details and/or formations for a specific match.                                                                                                                                        | ✅ Complete |
| `wyscout_advanced_stats`     | A highly advanced, context-driven tool for all advanced statistics, logically separated for queries about a *match*, a *player*, or a *team*.                                                                  | ✅ Complete |
| `wyscout_match_events`       | Retrieves the full, granular event stream for a match and includes powerful **client-side filtering** capabilities to analyze specific scenarios (e.g., all shots by a player).                               | ✅ Complete |
| `wyscout_video_tool`         | A safety-oriented tool for video. Provides "safe" methods to check for available qualities/offsets and an explicit "costly" method to generate video links that consumes usage minutes.                       | ✅ Complete |
| `wyscout_area_list`          | Retrieves a comprehensive list of all geographic areas, smartly combining live API results with the documented static list of custom regions (e.g., Europe, England, Scotland) for maximum reliability.        | ✅ Complete |
| `wyscout_round_info`         | Retrieves detailed information for a specific competition round (e.g., group stage, knockout phase, finals).                                                                                                 | ✅ Complete |
| `wyscout_referee_info`       | Retrieves detailed information for a specific referee.                                                                                                                                                       | ✅ Complete |
| `wyscout_coach_info`         | Retrieves detailed information for a specific coach.                                                                                                                                                         | ✅ Complete |

## 4. Design Philosophy

* **Unified Tools Over Fragmentation**: Instead of creating 5-10 small tools for each resource (e.g., `getPlayerDetails`, `getPlayerCareer`), we create one powerful tool (`wyscout_player_info`). This simplifies the agent's decision-making process. The agent decides *what resource* it needs (a player), and the tool's parameters handle *what specific data* about that resource is required.
* **Asynchronous & Parallel by Default**: All tools are built with `asyncio` and `aiohttp`. When a tool needs to fetch multiple pieces of information (e.g., a team's squad and fixtures), it makes these API calls concurrently, dramatically improving performance.
* **Schema-Driven Development**: We use `pydantic` extensively to define advanced input schemas. This includes using `Literal` types for type safety, clear descriptions for every parameter, and custom validators to enforce logical consistency (e.g., "you must provide `wyId` OR `areaId`, but not both"). This makes the tools self-documenting and less prone to misuse.
* **Safety and Clarity First**: For sensitive endpoints like video link generation, the tool's design makes it explicit which actions are "safe" (informational) and which are "costly" (consume API credits).

## 5. Getting Started (Preliminary)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/obinopaul/wyscout.git](https://github.com/obinopaul/wyscout.git)
    cd wyscout
    ```

2.  **Install dependencies:**
    (A `requirements.txt` file will be added soon.)
    ```bash
    pip install langchain pydantic aiohttp
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root of the project and add your Wyscout API credentials. The tools are designed to read these variables.
    ```
    WYSCOUT_API_TOKEN="YOUR_WYSCOUT_API_TOKEN_HERE"
    ```

4.  **Explore the Tools:**
    Navigate to the `tools/` directory to examine the code for each of the foundational tools.

## 6. Future Work

The completion of the foundational tools marks the end of Phase 1. The next phases of the project are even more exciting:

* **Phase 2: Agent Creation**
    * **Scout Agent**: Specializes in finding and retrieving detailed information about players and teams.
    * **Analyst Agent**: Specializes in retrieving and interpreting match events and advanced statistics.
    * **Archivist Agent**: Specializes in historical data, such as competition, season, and career information.
    * **Orchestrator Agent**: The master agent that receives user queries, understands the intent, and routes tasks to the appropriate specialist agent(s).

* **Phase 3: LangGraph Workflow Definition**
    * Define the state graph for the multi-agent system.
    * Implement the nodes (the agents) and the edges (the logic for how they pass information to each other).
    * Build in complex logic for loops and conditional routing (e.g., "if the `wyscout_id_search` tool returns multiple players, ask the user for clarification before proceeding").

* **Phase 4: User Interface & Application**
    * Build a user-friendly interface (e.g., a web app using Streamlit or FastAPI/React) that allows users to interact with the agent system.
    * Implement streaming to show the agent's "thought process" as it works.

## 7. Contributing

We welcome contributions! If you're interested in helping with this project, please feel free to open an issue or submit a pull request. Given the current phase, contributions related to agent creation and LangGraph workflow implementation are particularly welcome.

Please adhere to the existing code style and design philosophy when making contributions.

---

*This README is a living document and will be updated as the project evolves.*
