import os
from core.chatbot_utils import ChatbotBase
from core.chatbot_utils import LocalPortSaverTransport

import anthropic
import httpx
from dotenv import load_dotenv

class ClaudeHaiku(ChatbotBase):
    """
        Claude 3.5 Haiku chatbot (direct Anthropic API).
    """

    def __init__(self, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Call superclass
        super().__init__(remote_tls_port)

        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError('ANTHROPIC_API_KEY is not set in the environment variables.')

        # Create client that also saves the local port
        self._transport = LocalPortSaverTransport()
        # Use the Anthropic client but configure it to use our custom transport
        self._client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(transport=self._transport),
            base_url=f'https://api.anthropic.com:{remote_tls_port}'
        )

        # Store the model name
        self._model = 'claude-3-5-haiku-20241022'

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
            Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """

        # Send prompt
        response = []
        stream = self._client.messages.create(
            model=self._model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            max_tokens=2000,
            stream=True,
            temperature=temperature
        )
        
        for chunk in stream:
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                response.append(chunk.delta.text)

        # Return response
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
        return 'claude-3.5-haiku'

    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """

        # Match
        return 'api.anthropic.com' in server_name
