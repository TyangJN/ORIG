import os
import json
from datetime import datetime

class SearchResultsManager:
    def __init__(self, search_results_path, warm_up=False, session_id=None):
        self.search_results_path = search_results_path
        if warm_up:
            self.file_path = f"{search_results_path}/warm_up_results.json"
        else:
            self.file_path = f"{search_results_path}/all_results.json"
        self.session_id = session_id or f"search_{datetime.now().strftime('%Y%m%d_%H')}"

        # Ensure directory exists
        os.makedirs(search_results_path, exist_ok=True)

    def load_or_create_data(self):
        """Load existing data or create new file"""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}
    
    def save_data(self, data):
        """Save data to file"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def initialize_session(self, input_prompt):
        """Initialize or load session"""
        data = self.load_or_create_data()

        if self.session_id not in data:
            # Create new session
            data[self.session_id] = {
                "session_info": {
                    "session_id": self.session_id,
                    "input_prompt": input_prompt,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "total_rounds": 0,
                    "status": "initialized"
                },
                "rounds": [],
                "final_result": [],
            }
            self.save_data(data)
            print(f"Created new session: {self.session_id}")
        else:
            # Load existing session
            print(f"Loaded existing session: {self.session_id}")
            print(f"- Completed rounds: {len(data[self.session_id]['rounds'])}")
            print(f"- Status: {data[self.session_id]['session_info']['status']}")

        return data

    def add_round(self, round_id, round_plan, round_result, sub_queries_data, round_summary=None):
        """Add a complete round of search results"""
        data = self.load_or_create_data()

        # Build round data
        round_data = {
            "round_id": round_id,
            "round_plan": round_plan,
            "round_result": round_result,
            "sub_queries": sub_queries_data,
            "round_summary": round_summary or self._generate_round_summary(sub_queries_data)
        }

        # Check if round already exists (support overwrite)
        session_data = data[self.session_id]
        existing_index = None
        for i, existing_round in enumerate(session_data["rounds"]):
            if existing_round["round_id"] == round_id:
                existing_index = i
                break

        if existing_index is not None:
            session_data["rounds"][existing_index] = round_data
            print(f"Updated round {round_id} search results")
        else:
            session_data["rounds"].append(round_data)
            print(f"Added round {round_id} search results")

        # Update session info
        session_data["session_info"]["total_rounds"] = len(session_data["rounds"])
        session_data["session_info"]["last_updated"] = datetime.now().isoformat()
        session_data["session_info"]["status"] = "in_progress"

        self.save_data(data)
        return True

    def add_sub_queries(self, round_id, round_plan, round_result, sub_query_data):
        """Add single sub-question result (more granular)"""
        data = self.load_or_create_data()
        session_data = data[self.session_id]

        # Find target round
        target_round = None
        for round_data in session_data["rounds"]:
            if round_data["round_id"] == round_id:
                target_round = round_data
                break

        # If round doesn't exist, create new round
        if target_round is None:
            target_round = {
                "round_id": round_id,
                "round_plan": round_plan,
                "round_result": round_result,
                "sub_queries": [],
                "round_summary": {}
            }
            session_data["rounds"].append(target_round)

        # Check if sub-question already exists
        sub_query_id = sub_query_data["sub_query_id"]
        existing_index = None
        for i, sq in enumerate(target_round["sub_queries"]):
            if sq["sub_query_id"] == sub_query_id:
                existing_index = i
                break

        if existing_index is not None:
            target_round["sub_queries"][existing_index] = sub_query_data
            print(f"Updated sub-query: {sub_query_id}")
        else:
            target_round["sub_queries"].append(sub_query_data)
            print(f"Added sub-query: {sub_query_id}")

        # Update session status
        session_data["session_info"]["last_updated"] = datetime.now().isoformat()

        self.save_data(data)
        return True

    def finalize_session(self, final_result):
        """Complete search, add final result"""
        data = self.load_or_create_data()
        session_data = data[self.session_id]

        session_data["final_result"] = final_result
        session_data["session_info"]["status"] = "completed"
        session_data["session_info"]["completed_at"] = datetime.now().isoformat()

        self.save_data(data)
        print(f"Session completed: {self.session_id}")

    def get_current_progress(self):
        """Get current progress"""
        data = self.load_or_create_data()
        if self.session_id in data:
            session_data = data[self.session_id]
            return {
                "session_exists": True,
                "status": session_data["session_info"]["status"],
                "completed_rounds": len(session_data["rounds"]),
                "last_round_id": session_data["rounds"][-1]["round_id"] if session_data["rounds"] else 0,
                "last_updated": session_data["session_info"]["last_updated"]
            }
        return {"session_exists": False}

    def _generate_round_summary(self, sub_queries_data):
        """Generate round summary"""
        return {
            "sub_queries_count": len(sub_queries_data),
            "text_searches": len([sq for sq in sub_queries_data if sq.get("search_type") == "text"]),
            "image_searches": len([sq for sq in sub_queries_data if sq.get("search_type") == "image"]),
            "timestamp": datetime.now().isoformat()
        } 


