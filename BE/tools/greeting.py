from smolagents import tool,Tool
from langdetect import detect

class GreetingTools(Tool):
	"""A simple Greeting tools to let agent help to make a Greeting"""
	inputs = {
            "message": {
                "type": "string",
                "description": "The message to respond to and need to use  greetingWithEnglish or greetingWithZh base on user language" 
            }
    }
	output_type = "string"
	name = "greeting"
	description = "A tool to generate greetings base on user language"

	def __init__(self,systemPrompt:str):
		self.systemPrompt = systemPrompt
		self.is_initialized = True  

	def __str__(self):
		return f'Greeting(systemPrompt={self.systemPrompt})'
	def __call__(self, *args, **kwds):
		print(f'{self},{args},{kwds}')
		return f'Greeting called with arg={args},kwds={kwds}'
	def _detect_lang(self,text:str)->str:
		"""
		use langdetect to detect lang
		"""
		try:
			lang=detect(text)
			return lang
		except:
			return 'en'

	
	def forward(self, message: str) -> str:
		lang = self._detect_language(message)
		if lang == 'zh-tw':
			return self.greetingWithZh(message)
		else:
			return self.greetingWithEnglish(message)
	
	def greetingWithEnglish(self:object,message:str=None):
		"""Greeting when user enters English message."""
		return greetingWithEN(message=message)
	def greetingWithZh(self:object,message:str=None):
		"""Greeting when user enters zh-tw message. if use zh-cn no reply"""
		return greetingWithZH(message=message)


@tool
def greetingWithEN(message: str) -> str:
    """Greeting when user enters English message.
    Args:
        message: The message to respond to.

    Returns:
        str: A greeting response in English.
    """
    return f'Hello! {message}'


@tool
def greetingWithZH(message: str) -> str:
    """Greeting when user enters zh-tw message. And not use zh-cn
    Args:
        message: The message to respond to.

    Returns:
        str: A greeting response in zh-tw.
    """
    return f'你好! 收到了 告訴我你要幹嘛！{message}'
