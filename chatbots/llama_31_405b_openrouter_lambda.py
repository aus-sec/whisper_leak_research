import os
from core.chatbot_utils import ChatbotBase
from core.chatbot_utils import LocalPortSaverTransport

from openai import OpenAI
import httpx
from dotenv import load_dotenv

class Llama405BLambda(ChatbotBase):
    """
        Llama 3.1 405B Instruct over OpenRouter chatbot.
    """

    def __init__(self, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Call superclass
        super().__init__(remote_tls_port)

        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError('OPENROUTER_API_KEY is not set in the environment variables.')

        # Setting up the proper headers for OpenRouter
        headers = {
            'HTTP-Referer': 'Your-App-Name',  # Replace with your app name
            'X-Title': 'Your-App-Name'        # Replace with your app name
        }

        # Create client that also saves the local port
        self._transport = LocalPortSaverTransport()
        self._client = OpenAI(
            base_url=f'https://openrouter.ai:{remote_tls_port}/api/v1',
            api_key=api_key,
            http_client=httpx.Client(transport=self._transport, headers=headers)
        )

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
            Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """

        # Send prompt
        response = []
        
        # Use the correct model ID for DeepSeek R1
        stream = self._client.chat.completions.create(
            model='meta-llama/llama-3.1-405b-instruct',  # Use the correct model ID
            messages=[
                { 'role': 'user', 'content': prompt }
            ],
            stream=True,
            max_tokens=4000,
            temperature=temperature,
            # Specify the provider routing in extra_body
            extra_body={
                'provider': {
                    'order': [ 'Lambda' ],
                    'allow_fallbacks': False
                },
            }
        )
        
        # Get chunks
        for chunk in stream:
            assert chunk.provider == 'Lambda', Exception(f'Unexpected provider: {chunk.provider}')
            if hasattr(chunk, 'choices') and chunk.choices is not None and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content is not None:
                    response.append(chunk.choices[0].delta.content)
        
        # Return response
        return (response, self._transport.get_local_port())

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # For now we just return the default of 0.7 for DeepSeek R1
        return 0.7

    def get_common_name(self):
        """
            Gets the common name of the model.
        """

        # Return common name
        return 'llama-3.1-405b-instruct (OpenRouter Lambda)'

    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """

        # Match
        return 'openrouter.ai' in server_name
