# A simple class file to connect to OpenAI model hosted on Azure
# Parag Jain. 

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel

class PatientInformation(BaseModel):
    name: str
    mrn: str
    dates_of_service: str

class ClinicalFindings(BaseModel):
    admission_diagnosis: str
    procedure: str
    outcome: str

class InsurancePolicy(BaseModel):
    policy_number: str
    plan_type: str
    effective_date: str
    annual_deductible: str

class InvestigativeReport(BaseModel):
    purpose: str
    findings: str

class ClaimReviewSummary(BaseModel):
    patient_information: PatientInformation
    clinical_findings: ClinicalFindings
    insurance_policy: InsurancePolicy
    investigative_report: InvestigativeReport
    claim_verdict: str
    justification: str

class OutputSchema(BaseModel):
    claim_review_summary: ClaimReviewSummary

class AzureAIConnectionError(Exception):
    """Custom exception for Azure AI connection errors."""
    pass

class AzureAIConnection:
    def __init__(self):
        """
        AzureAIConnection class.
        Load environment variables and create an instance of the AzureOpenAI client.
        Requires the following environment variables:
        - AZURE_OPENAI_VERSION
        - AZURE_OPENAI_KEY
        - AZURE_OPENAI_BASE
        - AZURE_OPENAI_DEPLOYMENT_NAME
        """
        load_dotenv()
        try:
            self.client = self._initialize_client()
            self.deployment_name = self._get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME")
        except Exception as e:
            raise AzureAIConnectionError(e)

    def _get_env_variable(self, var_name: str) -> str:
        value = os.getenv(var_name)
        if not value:
            raise AzureAIConnectionError(f"Missing environment variable: {var_name}")
        return value

    def _initialize_client(self) -> AzureOpenAI:
        api_key = self._get_env_variable("AZURE_OPENAI_KEY")
        api_version = self._get_env_variable("AZURE_OPENAI_VERSION")
        azure_endpoint = self._get_env_variable("AZURE_OPENAI_BASE")

        print(f"Connecting to Azure OpenAI with the following settings:")
        print(f"API Version: {api_version}")
        print(f"Azure Endpoint: {azure_endpoint}")
        print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else api_key}")
        print(f"Deployment Name: {self._get_env_variable('AZURE_OPENAI_DEPLOYMENT_NAME')}")
        
        return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)

    def get_client(self) -> AzureOpenAI:
        """
        Return the AzureOpenAI client instance.
        Returns:
            AzureOpenAI: The client instance for Azure OpenAI.
        """
        if not self.client:
            raise AzureAIConnectionError("The Azure OpenAI client is not initialized.")
        return self.client

    def get_deployment_name(self) -> str:
        """
        Return the Azure deployment name.
        Returns:
            str: The deployment name for Azure OpenAI.
        """
        if not self.deployment_name:
            raise AzureAIConnectionError("The deployment name is not initialized.")
        return self.deployment_name

    def generate_response(self, prompt: str, system_content: str = "You are a helpful assistant.", max_tokens: int = 4000) -> str:
        """
        Generate a response from the Azure OpenAI model.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_content (str): The system content to define the assistant's role.
            max_tokens (int): The maximum number of tokens in the response.
        
        Returns:
            str: The response from the model.
        """
        
        try:
            print(f"Making request to deployment: {self.deployment_name}")
            print(f"Full endpoint URL: {self._get_env_variable('AZURE_OPENAI_BASE')}")
            print(f"API Version: {self._get_env_variable('AZURE_OPENAI_VERSION')}")
            
            # Try chat completions first (for GPT-3.5-turbo, GPT-4, etc.)
            try:
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                if not response.choices or len(response.choices) == 0:
                    raise AzureAIConnectionError("Empty response received from Azure OpenAI")
                
                return response.choices[0].message.content.strip()
            
            except Exception as chat_error:
                print(f"Chat completions failed: {chat_error}")
                print("Trying legacy completions API...")
                
                # Fallback to completions API (for text-davinci models)
                response = self.client.completions.create(
                    model=self.deployment_name,
                    prompt=f"System: {system_content}\n\nUser: {prompt}\n\nAssistant:",
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stop=None
                )
                if not response.choices or len(response.choices) == 0:
                    raise AzureAIConnectionError("Empty response received from Azure OpenAI")
                
                return response.choices[0].text.strip()
                
        except Exception as e:
            print(f"Full error details: {e}")
            raise AzureAIConnectionError(f"Error generating response: {str(e)}")

    def generate_structured_response(self, prompt: str, system_content: str = "You are an expert in processing of Insurance Claims Processes", max_tokens: int = 4000) -> str:
        """
        Generate a structured response from the Azure OpenAI model using the defined schema.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_content (str): The system content to define the assistant's role.
            max_tokens (int): The maximum number of tokens in the response.
        
        Returns:
            str: The structured response from the model in JSON format.
        """
        
        # Add schema instructions to the system content
        schema_instruction = f"""
        You must respond with a JSON object that follows this exact structure:
        {{
            "claim_review_summary": {{
                "patient_information": {{
                    "name": "patient name",
                    "mrn": "medical record number",
                    "dates_of_service": "service dates"
                }},
                "clinical_findings": {{
                    "admission_diagnosis": "diagnosis",
                    "procedure": "procedure performed",
                    "outcome": "patient outcome"
                }},
                "insurance_policy": {{
                    "policy_number": "policy number",
                    "plan_type": "insurance plan type",
                    "effective_date": "policy effective date",
                    "annual_deductible": "deductible amount"
                }},
                "investigative_report": {{
                    "purpose": "investigation purpose",
                    "findings": "key findings"
                }},
                "claim_verdict": "verdict (approved/denied/pending)",
                "justification": "detailed justification"
            }}
        }}
        
        Always return valid JSON format. {system_content}
        """
        
        return self.generate_response(prompt, schema_instruction, max_tokens)


# ## Test the class file. Uncomment this block if you want to test the class file by running this file.
# if __name__ == "__main__":
#     try:
#         azure_ai_connection = AzureAIConnection()
#         response = azure_ai_connection.generate_response(
#             prompt="Write a long essay on the history of the world"
#         )
#         print("Response from Azure OpenAI:", response)
#     except AzureAIConnectionError as e:
#         print(f"Failed to connect to Azure OpenAI: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")