import os
import re
import json
import asyncio
from dotenv import load_dotenv
from backend.services.report_service import COMMON_LAB_PATTERNS

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are Personal AI Doctor, a highly capable, empathetic, and responsible Personal AI Health Assistant.

CRITICAL RULES:
1. UNDERSTAND THE ACTUAL QUESTION: Before answering, analyze WHAT exactly the user is asking. Identify the specific entity, topic, or value they want to know about. Answer ONLY what was asked.

2. RESOLVE REFERENCES: When the user uses words like "it", "that", "this", "my report", "my result", "my problem", "this value", "previous one" — look at the conversation history to understand what they are referring to. Never guess.

3. CONVERSATION MEMORY: You must remember the full conversation context. If the user previously mentioned "headache", and now says "it is on the right side", "it" refers to the headache. Track the current topic and entity.

4. TOPIC CHANGES: If the user changes the topic (e.g., from "headache" to "blood sugar"), follow the new topic. Do not keep answering about the old topic.

5. PERSONAL vs GENERAL QUESTIONS: Distinguish between:
   - "What is hemoglobin?" → General explanation
   - "What is MY hemoglobin?" → User's actual report value
   These are DIFFERENT questions. Answer accordingly.

6. USE ACTUAL DATA FIRST: When the user asks about their report, weight, symptoms, or any personal data, use the ACTUAL VALUES from the provided context. Never give a generic explanation when specific data is available.

7. ZERO HALLUCINATION: NEVER invent patient data, test values, symptoms, medications, or diagnoses. If information is not available, say: "I don't have that information available."

8. MEDICAL SAFETY: Do not diagnose with certainty. Say "these symptoms can have several causes" rather than "you have X." Consider patient's age, allergies, medications, and conditions before giving advice.

9. BREVITY: Answer the question directly first. Only add extra context if useful. Do not dump long explanations unless the user asks for detail.

10. NATURAL CONVERSATION: Be conversational, warm, and empathetic. Address the patient by name naturally (not in every sentence). Greet when greeted. Answer small talk naturally.

11. LANGUAGE: Respond strictly in {language_instruction}. Do not switch languages unless requested.

12. REPORT DATA OVERRIDE: When the user asks about a specific test value (e.g., "What is my hemoglobin?"), use the ACTUAL report value provided in the context. The report data MUST override any general knowledge.

Patient Name: {patient_name}
"""

def normalize_language_instruction(language: str) -> str:
    lang = (language or "english").lower().strip()
    if lang in ["hi", "hindi", "हिंदी"]:
        return "Hindi (strictly in Hindi language using Devanagari script)"
    elif lang in ["hinglish", "hin-eng"]:
        return "Hinglish (Hindi language written in Roman/English script)"
    else:
        return "English"

class LLMService:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.default_model = os.getenv("DEFAULT_MODEL", "gemma4:e2b").strip()
        # OLLAMA_BASE_URL is the canonical env var (configurable for Render deployment)
        self.ollama_host = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")).strip()

    @property
    def model_name(self) -> str:
        if self.gemini_key:
            return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        elif self.openai_key:
            return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            return self.default_model

    @property
    def active_provider(self) -> str:
        if self.gemini_key:
            return "Gemini API"
        elif self.openai_key:
            return "OpenAI API"
        elif self.ollama_host:
            return f"Ollama Local ({self.default_model})"
        return "Unknown"

    def _build_messages(self, history, profile, active_diagnosis, intake_data, language, report_data=None, conversation_id=None):
        lang_instr = normalize_language_instruction(language)
        patient_name = profile.get("name") if (profile and profile.get("name") and profile.get("name") != "Guest User") else "Patient"
        system_content = SYSTEM_PROMPT_TEMPLATE.format(language_instruction=lang_instr, patient_name=patient_name)

        # Determine intent from the latest user message
        latest_user_msg = ""
        if history:
            # Find last user role message
            for msg in reversed(history):
                if msg.get("role") == "user":
                    latest_user_msg = msg.get("content", "")
                    break
        intent = self._extract_intent(latest_user_msg)

        context_blocks = []

        # 1. Patient Profile Context (only if meaningful data exists)
        if profile:
            p_name = profile.get("name", "Guest")
            p_age = profile.get("age", "Unspecified")
            p_gender = profile.get("gender", "Unspecified")
            p_height = profile.get("height", "Not specified")
            p_weight = profile.get("weight", "Not specified")
            p_allergies = profile.get("allergies", "None")
            p_conditions = profile.get("chronic_conditions", "None")

            profile_parts = []
            if p_name and p_name != "Guest User":
                profile_parts.append(f"Name: {p_name}")
            if p_age and p_age != "Unspecified":
                profile_parts.append(f"Age: {p_age}")
            if p_gender and p_gender != "Unspecified":
                profile_parts.append(f"Gender: {p_gender}")
            if p_height and p_height not in ("Not specified", "None", ""):
                profile_parts.append(f"Height: {p_height}")
            if p_weight and p_weight not in ("Not specified", "None", ""):
                profile_parts.append(f"Weight: {p_weight}")
            if p_allergies and p_allergies not in ("None", "none", ""):
                profile_parts.append(f"Allergies: {p_allergies}")
            if p_conditions and p_conditions not in ("None", "none", ""):
                profile_parts.append(f"Chronic Conditions: {p_conditions}")

            if profile_parts:
                context_blocks.append("PATIENT PROFILE:\n" + "\n".join(f"- {p}" for p in profile_parts))

        # 2. Active Intake & Assessment Context (only if present)
        if intake_data or active_diagnosis:
            intake_parts = []
            if intake_data and isinstance(intake_data, dict):
                syms = intake_data.get("symptoms", [])
                if syms:
                    if isinstance(syms, list):
                        syms = ", ".join(syms)
                    intake_parts.append(f"Reported Symptoms: {syms}")
                duration = intake_data.get("duration")
                if duration:
                    intake_parts.append(f"Duration: {duration}")
                trajectory = intake_data.get("trajectory")
                if trajectory:
                    intake_parts.append(f"Trajectory: {trajectory}")
                food = intake_data.get("recent_food_activity")
                if food and food.lower() != "none":
                    intake_parts.append(f"Recent Food/Activity: {food}")
                meds = intake_data.get("current_medications")
                if meds and meds.lower() != "none":
                    intake_parts.append(f"Current Medications: {meds}")
                severity = intake_data.get("severity_score")
                if severity:
                    intake_parts.append(f"Pain Severity: {severity}/10")
                notes = intake_data.get("notes")
                if notes:
                    intake_parts.append(f"Additional Notes: {notes}")

            if active_diagnosis and isinstance(active_diagnosis, dict):
                cond = active_diagnosis.get("condition")
                if cond:
                    intake_parts.append(f"Assessed Condition: {cond}")
                cause = active_diagnosis.get("cause")
                if cause:
                    intake_parts.append(f"Assessment Reasoning: {cause}")

            if intake_parts:
                context_blocks.append("ACTIVE CLINICAL CONTEXT:\n" + "\n".join(f"- {p}" for p in intake_parts))

        # 3. Medical Report Context (only if available and relevant)
        from backend.services.report_service import format_structured_report_context
        from backend.db.database import get_latest_patient_report

        fetched_report = report_data or (get_latest_patient_report(conversation_id) if conversation_id else get_latest_patient_report())
        if fetched_report:
            # If intent indicates a specific lab test, filter report context
            if intent and intent.get("type") == "lab_test":
                report_str = self._format_report_for_test(fetched_report, intent.get("entity"))
            else:
                report_str = format_structured_report_context(fetched_report)
            context_blocks.append(report_str)

        messages = [{"role": "system", "content": system_content}]
        if context_blocks:
            messages.append({"role": "system", "content": "\n\n".join(context_blocks)})

        # 4. Conversation History (last 20 messages for context)
        if history:
            for msg in history[-20:]:
                role = "assistant" if msg.get("role") in ("assistant", "ai") else "user"
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        # Log the message structure (safely, no private data)
        print(f"[LLMService] Messages sent: {len(messages)} | Model: {self.model_name} | Provider: {self.active_provider}")

        return messages

    async def generate_response_stream(self, conversation_id=None, history=None, profile=None,
                                       active_diagnosis=None, intake_data=None, language="english", report_data=None):
        messages = self._build_messages(history, profile, active_diagnosis, intake_data, language, report_data, conversation_id=conversation_id)

        # 1. Try Google Gemini API if key exists
        if self.gemini_key:
            try:
                async for chunk in self._stream_gemini(messages):
                    yield chunk
                return
            except Exception as e:
                print(f"[LLMService] Gemini API Error: {e}. Falling back...")

        # 2. Try OpenAI API if key exists
        if self.openai_key:
            try:
                async for chunk in self._stream_openai(messages):
                    yield chunk
                return
            except Exception as e:
                print(f"[LLMService] OpenAI API Error: {e}. Falling back...")

        # 3. Try Ollama local model (PRIMARY — always try)
        if self.ollama_host:
            try:
                async for chunk in self._stream_ollama(messages):
                    yield chunk
                return
            except Exception as e:
                print(f"[LLMService] Ollama Error: {e}")
                import traceback
                traceback.print_exc()

        # 4. If nothing works, return a clear error message
        yield "I'm sorry, the AI service is temporarily unavailable. Please try again in a moment."

    async def _stream_gemini(self, messages):
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        prompt_parts = []
        for msg in messages:
            role_prefix = "System" if msg["role"] == "system" else ("User" if msg["role"] == "user" else "Model")
            prompt_parts.append(f"{role_prefix}: {msg['content']}")
        full_prompt = "\n\n".join(prompt_parts)

        model = genai.GenerativeModel(model_name)
        response = await asyncio.to_thread(model.generate_content, full_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def _stream_openai(self, messages):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.openai_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
        async for part in stream:
            delta = part.choices[0].delta
            if delta.content:
                yield delta.content

    async def _stream_ollama(self, messages):
        from ollama import AsyncClient
        async_client = AsyncClient(host=self.ollama_host)
        model_name = self.default_model

        options = {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": 1024,
        }

        print(f"[LLMService] Calling Ollama model={model_name} with {len(messages)} messages")

        stream = await async_client.chat(
            model=model_name,
            messages=messages,
            options=options,
            stream=True,
        )

        in_think_block = False
        async for part in stream:
            message = part.get("message", {}) if isinstance(part, dict) else getattr(part, "message", {})
            chunk = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")

            if not chunk:
                thinking = message.get("thinking", "") if isinstance(message, dict) else getattr(message, "thinking", "")
                if thinking and not chunk:
                    continue

            if chunk:
                if "<think>" in chunk:
                    in_think_block = True
                    chunk = re.sub(r"<think>.*?(?:</think>|$)", "", chunk, flags=re.DOTALL)
                elif "</think>" in chunk:
                    in_think_block = False
                    chunk = re.sub(r".*?</think>", "", chunk, flags=re.DOTALL)
                elif in_think_block:
                    continue

                if chunk:
                    yield chunk


    def _extract_intent(self, user_msg: str) -> dict:
        """Very light intent extraction for lab test queries.
        Returns a dict like {'entity': 'Hemoglobin', 'type': 'lab_test'} or empty dict.
        """
        if not user_msg:
            return {}
        lowered = user_msg.lower()
        # Check for generic personal data requests
        for pattern in COMMON_LAB_PATTERNS:
            name = pattern["name"].lower()
            # simple keyword match
            if name in lowered:
                return {"entity": pattern["name"], "type": "lab_test"}
        # fallback for symptoms (not exhaustive)
        symptom_keywords = ["headache", "pain", "fever", "cough", "dizzy", "tired"]
        for sym in symptom_keywords:
            if sym in lowered:
                return {"entity": sym, "type": "symptom"}
        return {}

    def _format_report_for_test(self, report_record: dict, test_name: str) -> str:
        """Return a minimal report string containing only the requested test.
        Uses format_structured_report_context logic but filters to a single test.
        """
        from backend.services.report_service import format_structured_report_context
        extracted = report_record.get("extracted_json", {})
        if isinstance(extracted, str):
            try:
                extracted = json.loads(extracted)
            except Exception:
                extracted = {}
        tests = extracted.get("tests", [])
        # Find matching test (case‑insensitive)
        matching = [t for t in tests if t.get("test", "").lower() == test_name.lower()]
        if matching:
            # Build a custom context string similar to format_structured_report_context
            rep_name = report_record.get("report_name") or extracted.get("report_name") or "Medical Lab Report"
            rep_date = report_record.get("report_date") or extracted.get("date") or "Recent"
            lines = ["MEDICAL REPORT CONTEXT:", f"- Report Name: {rep_name}", f"- Date: {rep_date}", "- Parsed Lab Test Findings:"]
            for t in matching:
                t_name = t.get("test", "Test")
                val = t.get("value", "N/A")
                unit = t.get("unit", "")
                ref = t.get("reference_range", "")
                flag = t.get("flag", "normal")
                lines.append(f"  * {t_name}: {val} {unit} (Ref Range: {ref}) [Status: {flag.upper()}]")
            return "\n".join(lines)
        # If no specific test found, fallback to full report
        return format_structured_report_context(report_record)

llm_service = LLMService()
