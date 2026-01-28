import os
from core.chatbot_utils import ChatbotBase
from core.chatbot_utils import LocalPortSaverTransport

from openai import AzureOpenAI
import httpx
from dotenv import load_dotenv

import os

class Phi35MiniMoEInstruct(ChatbotBase):
    """
        Azure Phi-3.5-MoE-instruct model.
    """

    def __init__(self, remote_tls_port=443):
        """
        Creates an instance.
        """
        # Call superclass
        super().__init__(remote_tls_port)

        # Load environment variables from .env file
        load_dotenv()

        # Validate environment variables
        if not os.getenv('AZURE_PHI3_MOE_ENDPOINT'):
            raise ValueError('AZURE_PHI3_MOE_ENDPOINT is not set in the environment variables.')
        if not os.getenv('AZURE_PHI3_MOE_API_KEY'):
            raise ValueError('AZURE_PHI3_MOE_API_KEY is not set in the environment variables.')

        # Create custom transport to save local port
        self._transport = LocalPortSaverTransport()
        
        # Create OpenAI client with Azure configuration
        self._client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_PHI3_MOE_ENDPOINT'),
            api_key=os.getenv('AZURE_PHI3_MOE_API_KEY'),
            api_version="2024-05-01-preview",
            http_client=httpx.Client(transport=self._transport)
        )

    def send_prompt(self, prompt, temperature):
        """
        Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
        Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """
        # Send prompt
        response = []

        # Use the OpenAI streaming API
        stream = self._client.chat.completions.create(
            model="Phi-3.5-MoE-instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stream=True
        )

        # Process the streaming response
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response.append(chunk.choices[0].delta.content)
        
        # Return response and local port
        return (response, self._transport.get_local_port())

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # For now we just return the default of 1.0
        return 1.0

    def get_common_name(self):
        """
            Gets the common name of the model.
        """

        # Return common name
        return 'Phi-3.5-MoE-instruct'

    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """

        # Match
        return 'services.ai.azure.com' in server_name