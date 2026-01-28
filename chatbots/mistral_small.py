import os
from core.chatbot_utils import ChatbotBase
from core.chatbot_utils import LocalPortSaverTransport

from mistralai import Mistral
import httpx
from dotenv import load_dotenv
import asyncio

class MistralSmall(ChatbotBase):
    """
        Mistral Small chatbot.
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
        key = os.getenv('MISTRAL_API_KEY')
        if not key:
            raise ValueError('MISTRAL_API_KEY is not set in the environment variables.')

        # Create client that also saves the local port
        self._transport = LocalPortSaverTransport()
        self._client = Mistral(
            api_key=key,
            client=httpx.Client(transport=self._transport)
        )

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
            Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """

        # Make sure asyncio has a loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Send prompt
        response = []
        stream = self._client.chat.stream(
            model='mistral-small-2503',
            messages=[ { 'role': 'user', 'content': prompt } ],
            stream=True,
            temperature=temperature
        )
        for chunk in stream:
            if len(chunk.data.choices) > 0:
                response.append(chunk.data.choices[0].delta.content)

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
        return 'mistral-small-2503'

    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """

        # Match
        return 'api.mistral.ai' in server_name
