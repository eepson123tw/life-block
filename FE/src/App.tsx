import { useState, useEffect, useRef } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Loader2, AlertTriangle, User, Bot, Send, Info } from "lucide-react"; // Added Info icon
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select" // Import Select components
import { cn } from "@/lib/utils";
import { gsap } from 'gsap';

// --- Types ---
type Message = {
  id: string;
  role: 'user' | 'ai' | 'error';
  content: string;
};

type FormData = {
  name: string;
  age: string;
  gender: string;
  country: string;
};

// --- Constants ---
const GENDERS = ["Male", "Female", "Non-binary", "Prefer not to say"];
// Basic country list - expand as needed
const COUNTRIES = ["USA", "Canada", "Mexico", "UK", "Germany", "France", "Japan", "China", "India", "Brazil", "Australia", "Taiwan"];
// Simple language to country mapping (expandable)
const LANG_TO_COUNTRY_MAP: Record<string, string> = {
  "en": "USA", // Default English to USA
  "en-US": "USA",
  "en-GB": "UK",
  "en-CA": "Canada",
  "en-AU": "Australia",
  "fr": "France",
  "de": "Germany",
  "es": "Mexico", // Default Spanish to Mexico (adjust as needed)
  "es-MX": "Mexico",
  "ja": "Japan",
  "zh": "China", // Default Chinese to China
  "zh-CN": "China",
  "zh-TW": "Taiwan", // Specific locale for Taiwan
  "hi": "India",
  "pt": "Brazil", // Default Portuguese to Brazil
  "pt-BR": "Brazil",
};

// --- Background Component (Unchanged) ---
const AnimatedBackground = () => {
  const bgRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!bgRef.current) return;
    const tl = gsap.timeline({ repeat: -1, yoyo: true });
    tl.to(bgRef.current, { duration: 45, backgroundPosition: "100% 100%", ease: "linear" });
    return () => { tl.kill(); };
  }, []);
  return (
    <div
      ref={bgRef}
      className="fixed inset-0 -z-10 h-full w-full bg-gradient-to-br from-indigo-100 via-purple-100 to-pink-100 dark:from-indigo-900/80 dark:via-purple-900/80 dark:to-pink-900/80"
      style={{ backgroundSize: '200% 200%' }}
    />
  );
};

// --- Main App Component ---
function App() {
  // Form State
  const [formData, setFormData] = useState<FormData>({
    name: '',
    age: '',
    gender: '',
    country: ''
  });

  // Other State (kept from previous version)
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [count, setCount] = useState(0); // Counter kept

  // Refs (kept from previous version)
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll Effect (kept from previous version)
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  // --- Automatic Country Detection ---
  useEffect(() => {
    const detectedLang = navigator.language; // e.g., "en-US", "fr-FR", "zh-TW"
    const primaryLang = detectedLang.split('-')[0]; // e.g., "en", "fr", "zh"

    // Try specific locale first, then primary language
    const mappedCountry = LANG_TO_COUNTRY_MAP[detectedLang] || LANG_TO_COUNTRY_MAP[primaryLang];

    if (mappedCountry && COUNTRIES.includes(mappedCountry)) {
      setFormData(prev => ({ ...prev, country: mappedCountry }));
    }
    // You could add a default fallback here if needed
    // else { setFormData(prev => ({ ...prev, country: 'USA' })); }

  }, []); // Runs only on mount

  // Handle form input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Handle select changes (gender, country)
  const handleSelectChange = (name: keyof FormData) => (value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // --- Fetch Function (Modified) ---
  const handleFetch = async (stream: boolean, currentFormData: FormData) => {
    setIsLoading(true);

    // Construct the query text from form data
    const queryText = `help me with the User Profile - Name: ${currentFormData.name || 'N/A'}, Age: ${currentFormData.age || 'N/A'}, Gender: ${currentFormData.gender || 'N/A'}, Country: ${currentFormData.country || 'N/A'} and use tool to find the  Profile country average years old,and you need to reply lang with ${formData.country}`;

    // Add a user message summarizing the submitted data
    const userSummaryMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: `Submitted Profile:\nName: ${currentFormData.name}\nAge: ${currentFormData.age}\nGender: ${currentFormData.gender}\nCountry: ${currentFormData.country}`
    };
    setMessages(prev => [...prev, userSummaryMessage]);

    const endpoint = 'http://localhost:8000/query'; // Adjust if needed

    if (!stream) {
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          // Send the constructed queryText
          body: JSON.stringify({ query_text: queryText, stream: false }),
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        if (data.success) {
          setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'ai', content: data.response }]);
        } else {
          throw new Error(data.response || 'API returned an error');
        }
      } catch (err) {
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'error', content: `Failed to fetch: ${err instanceof Error ? err.message : String(err)}` }]);
      } finally {
        setIsLoading(false);
      }
    } else { // Streaming
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          // Send the constructed queryText
          body: JSON.stringify({ query_text: queryText, stream: true }),
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        if (!response.body) throw new Error('ReadableStream not supported');

        const aiMessageId = crypto.randomUUID();
        setMessages(prev => [...prev, { id: aiMessageId, role: 'ai', content: '' }]);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedChunks = '';

        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          accumulatedChunks += decoder.decode(value, { stream: true });
          setMessages(prev => prev.map(msg => msg.id === aiMessageId ? { ...msg, content: accumulatedChunks } : msg));
        }
        accumulatedChunks += decoder.decode(); // Final decode
        setMessages(prev => prev.map(msg => msg.id === aiMessageId ? { ...msg, content: accumulatedChunks } : msg));

      } catch (err) {
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'error', content: `Streaming failed: ${err instanceof Error ? err.message : String(err)}` }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  // --- Form Submission Handler ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isLoading) return;
    // Basic validation: check if at least name and country are filled? Adjust as needed.
    if (!formData.name || !formData.country) {
      // Optionally show an error message to the user
      alert("Please fill in at least Name and Country.");
      return;
    }
    await handleFetch(isStreaming, formData);
    // Optionally clear form after submission?
    // setFormData({ name: '', age: '', gender: '', country: '' });
  };

  return (
    <div className="relative flex flex-col min-h-screen">
      <AnimatedBackground />

      <div className="container mx-auto px-4 py-6 flex flex-col h-screen max-w-4xl z-10">
        {/* Header (Unchanged) */}
        <header className="flex items-center justify-center mb-6">
          <div className="flex items-center gap-4 bg-background/60 dark:bg-background/80 backdrop-blur-sm p-3 rounded-lg shadow-sm">
            <div className="flex gap-2">
              <a href="https://vite.dev" target="_blank" rel="noreferrer" className="transition-opacity hover:opacity-80">
                <img src={viteLogo} className="h-8 w-8" alt="Vite logo" />
              </a>
              <a href="https://react.dev" target="_blank" rel="noreferrer" className="transition-opacity hover:opacity-80">
                <img src={reactLogo} className="h-8 w-8 animate-spin-slow" alt="React logo" />
              </a>
            </div>
            <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 dark:from-indigo-400 dark:via-purple-400 dark:to-pink-400">
              Life-Block AI Assistant
            </h1>
          </div>
        </header>

        {/* Card Layout: Form on Left, Response on Right (for wider screens) */}
        {/* On smaller screens, stack them */}
        <div className="flex flex-col md:flex-row flex-1 gap-6 overflow-hidden">

          {/* Form Card */}
          <Card className="w-full md:w-1/3 border shadow-lg overflow-hidden bg-card/80 dark:bg-card/70 backdrop-blur-md flex flex-col">
            <CardHeader className="border-b py-3 px-4">
              <CardTitle className="text-lg font-medium text-primary dark:text-primary/90">Your Information</CardTitle>
            </CardHeader>
            {/* Form takes remaining space */}
            <form onSubmit={handleSubmit} className="flex flex-col flex-1">
              <CardContent className="flex-1 p-4 space-y-4 overflow-y-auto">
                {/* Name Input */}
                <div>
                  <Label htmlFor="name" className="text-sm font-medium text-muted-foreground">Name</Label>
                  <Input
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    placeholder="Enter your name"
                    disabled={isLoading}
                    required
                    className="mt-1 focus-visible:ring-purple-400"
                  />
                </div>
                {/* Age Input */}
                <div>
                  <Label htmlFor="age" className="text-sm font-medium text-muted-foreground">Your Age</Label>
                  <Input
                    id="age"
                    name="age"
                    type="number"
                    value={formData.age}
                    onChange={handleInputChange}
                    placeholder="e.g., 30"
                    disabled={isLoading}
                    min="0"
                    className="mt-1 focus-visible:ring-purple-400"
                  />
                </div>
                {/* Gender Select */}
                <div>
                  <Label htmlFor="gender" className="text-sm font-medium text-muted-foreground">Gender</Label>
                  <Select
                    name="gender"
                    value={formData.gender}
                    onValueChange={handleSelectChange('gender')}
                    disabled={isLoading}
                  >
                    <SelectTrigger id="gender" className="mt-1 focus:ring-purple-400">
                      <SelectValue placeholder="Select gender" />
                    </SelectTrigger>
                    <SelectContent>
                      {GENDERS.map(g => <SelectItem key={g} value={g}>{g}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                {/* Country Select */}
                <div>
                  <Label htmlFor="country" className="text-sm font-medium text-muted-foreground">Country</Label>
                  <Select
                    name="country"
                    value={formData.country}
                    onValueChange={handleSelectChange('country')}
                    disabled={isLoading}
                    required
                  >
                    <SelectTrigger id="country" className="mt-1 focus:ring-purple-400">
                      <SelectValue placeholder="Select country" />
                    </SelectTrigger>
                    <SelectContent>
                      {COUNTRIES.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
              {/* Form Footer */}
              <CardFooter className="p-4 border-t mt-auto">
                <div className='w-full space-y-3'>
                  {/* Submit Button */}
                  <Button
                    type="submit"
                    disabled={isLoading || !formData.name || !formData.country} // Example validation
                    className={cn(
                      "w-full text-white transition-colors duration-200",
                      isLoading || !formData.name || !formData.country
                        ? "bg-muted-foreground/50"
                        : "bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
                    )}
                  >
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Send className="h-4 w-4 mr-2" />}
                    Get AI Response
                  </Button>
                  {/* Options Row */}
                  <div className="flex flex-wrap items-center justify-between gap-4 text-xs">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="streaming-mode"
                        checked={isStreaming}
                        onCheckedChange={setIsStreaming}
                        disabled={isLoading}
                        aria-labelledby="streaming-label"
                        className="data-[state=checked]:bg-purple-500"
                      />
                      <Label htmlFor="streaming-mode" id="streaming-label" className="text-muted-foreground">
                        Stream response
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2 ml-auto">
                      <span className="text-muted-foreground">Counter:</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCount((c) => c + 1)}
                        className="h-7 px-2 font-mono border-muted-foreground/30 focus-visible:ring-purple-400"
                      >
                        {count}
                      </Button>
                    </div>
                  </div>
                </div>
              </CardFooter>
            </form>
          </Card>

          {/* Response Card */}
          <Card className="w-full md:w-2/3 border shadow-lg overflow-hidden bg-card/80 dark:bg-card/70 backdrop-blur-md flex flex-col">
            <CardHeader className="border-b py-3 px-4">
              <CardTitle className="text-lg font-medium text-primary dark:text-primary/90">AI Response</CardTitle>
            </CardHeader>
            {/* ScrollArea for responses */}
            <CardContent className="flex-1 p-0 overflow-hidden">
              <ScrollArea className="h-full" ref={scrollAreaRef}>
                <div className="p-4 md:p-6 space-y-6">
                  {/* Empty state for responses */}
                  {messages.length === 0 && !isLoading && (
                    <div className="flex flex-col items-center justify-center py-16 text-center">
                      <div className="rounded-full bg-gradient-to-tr from-indigo-200 to-purple-200 dark:from-indigo-800 dark:to-purple-800 p-4 mb-4 shadow-sm">
                        <Info className="h-6 w-6 text-indigo-600 dark:text-indigo-300" />
                      </div>
                      <p className="text-muted-foreground">Fill the form to get a response...</p>
                    </div>
                  )}

                  {/* Message display (similar to before, shows user summary & AI response) */}
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={cn(
                        "flex items-start gap-3 animate-in fade-in-0 slide-in-from-bottom-4 duration-300 ease-out",
                        message.role === 'user' ? "justify-end" : "justify-start"
                      )}
                    >
                      {/* AI/Error/User Icon */}
                      {message.role !== 'user' && ( // AI or Error Icon
                        <Avatar className={cn("h-8 w-8 border", message.role === 'error' ? "border-destructive/50" : "border-purple-300 dark:border-purple-700")}>
                          <AvatarFallback className={message.role === 'error' ? 'bg-destructive/10 text-destructive' : 'bg-purple-100 text-purple-600 dark:bg-purple-900/50 dark:text-purple-300'}>
                            {message.role === 'ai' ? <Bot size={16} /> : <AlertTriangle size={16} />}
                          </AvatarFallback>
                        </Avatar>
                      )}
                      {message.role === 'user' && ( // User Icon for summary
                        <Avatar className="h-8 w-8 border border-blue-300 dark:border-blue-600">
                          <AvatarFallback className="bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-300">
                            <User size={16} />
                          </AvatarFallback>
                        </Avatar>
                      )}

                      {/* Message Content */}
                      {message.role === 'error' ? ( // Error Alert
                        <Alert variant="destructive" className="max-w-[90%] my-1 shadow-sm">
                          <AlertTriangle className="h-4 w-4" />
                          <AlertTitle>Error</AlertTitle>
                          <AlertDescription className="text-sm whitespace-pre-wrap break-words">{message.content}</AlertDescription>
                        </Alert>
                      ) : ( // User Summary or AI Response Bubble
                        <div className={cn("p-3 rounded-lg max-w-[90%] shadow-sm",
                          message.role === 'user' ? "bg-blue-100 text-blue-900 dark:bg-blue-900 dark:text-blue-100 rounded-br-none ml-auto" // User summary style
                            : "bg-purple-100 text-purple-900 dark:bg-purple-900 dark:text-purple-100 rounded-bl-none" // AI style
                        )}>
                          <p className="text-sm whitespace-pre-wrap break-words">
                            {message.content || (message.role === 'ai' && isLoading && message.id === messages[messages.length - 1]?.id ? '...' : '')}
                          </p>
                        </div>
                      )}

                      {/* Right-aligned spacer for User message */}
                      {message.role === 'user' && <div className="w-8 h-8 flex-shrink-0"></div>}
                    </div>
                  ))}

                  {/* Loading Indicator (when fetching response) */}
                  {isLoading && messages[messages.length - 1]?.role !== 'ai' && (
                    <div className="flex items-start gap-3 animate-in fade-in-0">
                      <Avatar className="h-8 w-8 border border-purple-300 dark:border-purple-700">
                        <AvatarFallback className='bg-purple-100 text-purple-600 dark:bg-purple-900/50 dark:text-purple-300'><Bot size={16} /></AvatarFallback>
                      </Avatar>
                      <div className="bg-muted/70 p-3 rounded-lg rounded-bl-none">
                        <div className="flex space-x-1.5">
                          <div className="h-2 w-2 rounded-full bg-purple-400 dark:bg-purple-600 animate-bounce"></div>
                          <div className="h-2 w-2 rounded-full bg-purple-400 dark:bg-purple-600 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                          <div className="h-2 w-2 rounded-full bg-purple-400 dark:bg-purple-600 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Scroll anchor */}
                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

        </div> {/* End Form/Response Flex Container */}
      </div> {/* End Main Container */}
    </div> // End Relative Wrapper
  );
}

export default App;
