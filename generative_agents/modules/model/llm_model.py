"""generative_agents.model.llm_model"""

import time
import re
import requests


class LLMModel:
    def __init__(self, config):
        self._api_key = config["api_key"]
        self._base_url = config["base_url"]
        self._model = config["model"]
        self._meta_responses = []
        self._summary = {"total": [0, 0, 0]}

        self._handle = self.setup(config)
        self._enabled = True

    def setup(self, config):
        raise NotImplementedError(
            "setup is not support for " + str(self.__class__)
        )

    def completion(
        self,
        prompt,
        retry=10,
        callback=None,
        failsafe=None,
        caller="llm_normal",
        **kwargs
    ):
        response, self._meta_responses = None, []
        self._summary.setdefault(caller, [0, 0, 0])
        for attempt in range(retry):
            try:
                # On retry attempts, add stricter format instructions to prompt
                enhanced_prompt = prompt
                if attempt > 0 and ("schedule_daily" in caller or "schedule_decompose" in caller):
                    if "schedule_daily" in caller:
                        enhanced_prompt = prompt + "\n\nREMINDER: Output ONLY the schedule lines in format [HH:MM] Activity. No explanations."
                    elif "schedule_decompose" in caller:
                        enhanced_prompt = prompt + "\n\nREMINDER: Output ONLY the numbered subtask lines starting with '1) '. No explanations or answers."
                
                meta_response = self._completion(enhanced_prompt, **kwargs).strip()
                self._meta_responses.append(meta_response)
                self._summary["total"][0] += 1
                self._summary[caller][0] += 1
                if callback:
                    response = callback(meta_response)
                else:
                    response = meta_response
            except Exception as e:
                print(f"LLMModel.completion() caused an error (attempt {attempt + 1}/{retry}): {e}")
                time.sleep(5)
                response = None
                continue
            if response is not None:
                break
        pos = 2 if response is None else 1
        self._summary["total"][pos] += 1
        self._summary[caller][pos] += 1
        return response or failsafe

    def _completion(self, prompt, **kwargs):
        raise NotImplementedError(
            "_completion is not support for " + str(self.__class__)
        )

    def is_available(self):
        return self._enabled  # and self._summary["total"][2] <= 10

    def get_summary(self):
        des = {}
        for k, v in self._summary.items():
            des[k] = "S:{},F:{}/R:{}".format(v[1], v[2], v[0])
        return {"model": self._model, "summary": des}

    def disable(self):
        self._enabled = False

    @property
    def meta_responses(self):
        return self._meta_responses


class OpenAILLMModel(LLMModel):
    def setup(self, config):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _completion(self, prompt, temperature=0.5):
        messages = [{"role": "user", "content": prompt}]
        response = self._handle.chat.completions.create(
            model=self._model, messages=messages, temperature=temperature
        )
        if len(response.choices) > 0:
            return response.choices[0].message.content
        return ""


class OllamaLLMModel(LLMModel):
    def setup(self, config):
        # Get timeout from config, default to 300 seconds (5 minutes)
        self._timeout = config.get("timeout", 300)
        return None

    def ollama_chat(self, messages, temperature):
        headers = {
            "Content-Type": "application/json"
        }
        
        # Determine the correct endpoint
        # If base_url ends with /v1, use OpenAI-compatible endpoint
        # Otherwise, try OpenAI-compatible first, then fall back to native Ollama API
        base_url_clean = self._base_url.rstrip('/')
        
        # Try OpenAI-compatible endpoint first
        if base_url_clean.endswith('/v1'):
            endpoint = f"{base_url_clean}/chat/completions"
        else:
            endpoint = f"{base_url_clean}/v1/chat/completions"
        
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        try:
            response = requests.post(
                url=endpoint,
                headers=headers,
                json=params,
                stream=False,
                timeout=self._timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Check if it's a model not found error
            try:
                error_response = e.response.json()
                if "error" in error_response and "not found" in error_response.get("error", {}).get("message", "").lower():
                    error_msg = error_response.get("error", {}).get("message", "Unknown error")
                    raise ValueError(f"Model not found: {error_msg}. Please check if the model '{self._model}' is installed in Ollama.")
            except (ValueError, KeyError, AttributeError):
                pass
            
            # If 404 and we tried OpenAI-compatible endpoint, try native Ollama API
            if e.response.status_code == 404 and '/v1/chat/completions' in endpoint:
                # Try native Ollama API endpoints
                base_url_no_v1 = base_url_clean.replace('/v1', '').rstrip('/')
                
                # First try /api/chat (newer Ollama versions)
                endpoints_to_try = [
                    f"{base_url_no_v1}/api/chat",
                    f"{base_url_no_v1}/api/generate"
                ]
                
                for native_endpoint in endpoints_to_try:
                    try:
                        # Convert messages to prompt format for /api/generate
                        if "/api/generate" in native_endpoint:
                            # Convert messages to a single prompt string
                            prompt_parts = []
                            for msg in messages:
                                role = msg.get("role", "user")
                                content = msg.get("content", "")
                                if role == "system":
                                    prompt_parts.append(f"System: {content}")
                                elif role == "user":
                                    prompt_parts.append(f"User: {content}")
                                elif role == "assistant":
                                    prompt_parts.append(f"Assistant: {content}")
                            prompt = "\n".join(prompt_parts)
                            
                            ollama_params = {
                                "model": self._model,
                                "prompt": prompt,
                                "options": {
                                    "temperature": temperature,
                                },
                                "stream": False,
                            }
                        else:
                            # Use /api/chat format
                            ollama_params = {
                                "model": self._model,
                                "messages": messages,
                                "options": {
                                    "temperature": temperature,
                                },
                                "stream": False,
                            }
                        
                        response = requests.post(
                            url=native_endpoint,
                            headers=headers,
                            json=ollama_params,
                            stream=False,
                            timeout=self._timeout
                        )
                        response.raise_for_status()
                        ollama_response = response.json()
                        
                        # Convert Ollama response format to OpenAI-compatible format
                        content = None
                        if "/api/chat" in native_endpoint:
                            if "message" in ollama_response and "content" in ollama_response["message"]:
                                content = ollama_response["message"]["content"]
                        elif "/api/generate" in native_endpoint:
                            if "response" in ollama_response:
                                content = ollama_response["response"]
                        
                        if content:
                            return {
                                "choices": [{
                                    "message": {
                                        "content": content
                                    }
                                }]
                            }
                        else:
                            raise ValueError(f"Unexpected Ollama response format: {ollama_response}")
                    except requests.exceptions.HTTPError as inner_e:
                        if inner_e.response.status_code == 404 and native_endpoint != endpoints_to_try[-1]:
                            # Try next endpoint
                            continue
                        else:
                            raise
                # If all endpoints failed
                raise requests.exceptions.HTTPError(f"All Ollama API endpoints failed. Last error: {e}")
            else:
                raise

    def _completion(self, prompt, temperature=0.5):
        # Check for Qwen models (qwen2.5, qwen3, etc.)
        if ("qwen" in self._model.lower() or "qwen2.5" in self._model.lower() or "qwen3" in self._model.lower()) and "\n/nothink" not in prompt:
            # Disable think for Qwen3 model to improve inference speed
            prompt += "\n/nothink"
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.ollama_chat(messages=messages, temperature=temperature)
            # Check if response is valid and has the expected structure
            if not response:
                print(f"OllamaLLMModel._completion(): Empty response from API")
                return ""
            if "choices" not in response:
                print(f"OllamaLLMModel._completion(): Invalid response format. Response keys: {list(response.keys())}")
                print(f"OllamaLLMModel._completion(): Response content: {response}")
                return ""
            if not isinstance(response["choices"], list) or len(response["choices"]) == 0:
                print(f"OllamaLLMModel._completion(): No choices in response")
                return ""
            if "message" not in response["choices"][0] or "content" not in response["choices"][0]["message"]:
                print(f"OllamaLLMModel._completion(): Invalid choice structure: {response['choices'][0]}")
                return ""
            
            ret = response["choices"][0]["message"]["content"]
            # Filter out text within <think> or <think> tags from output to avoid affecting subsequent logic
            ret = re.sub(r"<think>.*?</think>", "", ret, flags=re.DOTALL)
            ret = re.sub(r"<think>.*?</think>", "", ret, flags=re.DOTALL)
            return ret
        except requests.exceptions.RequestException as e:
            print(f"OllamaLLMModel._completion(): Request error: {e}")
            raise
        except (KeyError, IndexError, TypeError) as e:
            print(f"OllamaLLMModel._completion(): Response parsing error: {e}")
            print(f"OllamaLLMModel._completion(): Response was: {response if 'response' in locals() else 'N/A'}")
            return ""


def create_llm_model(llm_config):
    """Create llm model"""

    if llm_config["provider"] == "ollama":
        return OllamaLLMModel(llm_config)

    elif llm_config["provider"] == "openai":
        return OpenAILLMModel(llm_config)
    else:
        raise NotImplementedError(
            "llm provider {} is not supported".format(llm_config["provider"])
        )
    return None


def parse_llm_output(response, patterns, mode="match_last", ignore_empty=False, failsafe=None):
    """
    Parse LLM output using regex patterns
    
    Args:
        response: LLM response string
        patterns: List of regex patterns or single pattern string
        mode: "match_first", "match_last", or "match_all"
        ignore_empty: If True, don't raise error on no match
        failsafe: Default value to return if no match found (instead of raising error)
        
    Returns:
        Matched string(s) or failsafe value
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    rets = []
    
    # Try matching each line
    for line in response.split("\n"):
        # Clean markdown formatting
        line_clean = line.replace("**", "").replace("*", "").strip()
        for pattern in patterns:
            if pattern:
                # Try original line first
                matchs = re.findall(pattern, line)
                if not matchs:
                    # Try cleaned line
                    matchs = re.findall(pattern, line_clean)
            else:
                matchs = [line_clean]
            if len(matchs) >= 1:
                rets.append(matchs[0])
                break
    
    # If no line-by-line matches, try matching the whole response
    if not rets:
        for pattern in patterns:
            if pattern:
                matchs = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
                if matchs:
                    rets.extend(matchs)
                    break
    
    # Handle no matches
    if not rets:
        # Print debug info when matching fails
        print(f"parse_llm_output: Failed to match. Response preview: {response[:500]}...")
        print(f"parse_llm_output: Patterns tried: {patterns}")
        
        # Use failsafe if provided
        if failsafe is not None:
            print(f"parse_llm_output: Using failsafe value: {failsafe}")
            return failsafe
        
        # If ignore_empty is True, return None instead of raising error
        if ignore_empty:
            return None
        
        # Otherwise, raise error (original behavior)
        raise ValueError(f"Failed to match llm output. Response: {response[:200]}... Patterns: {patterns}")
    
    # Return based on mode
    if mode == "match_first":
        return rets[0]
    if mode == "match_last":
        return rets[-1]
    if mode == "match_all":
        return rets
    return None
