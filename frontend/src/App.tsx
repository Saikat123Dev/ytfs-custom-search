import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

interface SearchResult {
  timestamp: number;
  snippet: string;
  url: string;
  score: number;
}

interface InputVideoBase {
  video_id: string;
  title: string;
  url: string;
  segments:SearchResult[]
}



interface RelatedVideo {
  video_id: string;
  title: string;
  channel_title: string;
  description: string;
  published_at: string;
  url: string;
  best_segment: SearchResult;
  has_relevant_content: boolean;
}

interface ApiResponseWithSuggestions {
  suggestions_enabled: true;
  input_video: InputVideoBase;
  total_input_segments: number;
  related_videos: RelatedVideo[];
  total_related_videos: number;
}

interface ApiResponseWithoutSuggestions {
  suggestions_enabled: false;
  input_video: InputVideoBase;
  total_input_segments: number;
}

function App() {
  const [videoLink, setVideoLink] = useState('')
  const [previewVideoId, setpreviewVideoId] = useState('')
  
  const [showPreview, setShowPreview] = useState(false)
  const [autoExtract, setAutoExtract] = useState(false)
  const [currentTabUrl, setCurrentTabUrl] = useState('')
  const [userInput, setUserInput] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [currentTime, setCurrentTime] = useState(0)
  const [enableSuggestions,setenableuggestions]=useState(false);
  const [keepExtensionOpen, setKeepExtensionOpen] = useState(false)

  const[Searchsuggestions,setSearchSuggestions]=useState<RelatedVideo[]>([])
  const iframeRef = useRef<HTMLIFrameElement>(null)
 

  const extractVideoId = (url: string) => {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
      /youtube\.com\/watch\?.*v=([^&\n?#]+)/
    ]
    
    
    for (const pattern of patterns) {
      const match = url.match(pattern)
      if (match) return match[1]
    }
    return null
  }
   const videoId = useMemo(() => extractVideoId(videoLink), [videoLink])
const isValidLink = useMemo(() => !!videoId, [videoId])

  const effectiveVideoId = previewVideoId || videoId

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const extractFromCurrentTab = async () => {
    try {
      if (typeof chrome !== 'undefined' && chrome.tabs) {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (tabs[0]) {
            const tabUrl = tabs[0].url
            setCurrentTabUrl(tabUrl ?? '')
            if (extractVideoId(tabUrl ?? '')) {
              setVideoLink(tabUrl ?? '')
            }
          }
        })
      } else {
        const currentUrl = window.location.href
        setCurrentTabUrl(currentUrl)
        if (extractVideoId(currentUrl)) {
          setVideoLink(currentUrl)
        }
      }
    } catch (err) {
      console.log('Tab access not available')
    }
  }

  useEffect(() => {
    const initializeApp = async () => {
      if (autoExtract) {
        await extractFromCurrentTab()
      }
    }
    
    initializeApp()
  }, [autoExtract])




  useEffect(() => {
    if (iframeRef.current && effectiveVideoId && currentTime > 0) {
      const newSrc = `https://www.youtube.com/embed/${effectiveVideoId}?start=${currentTime}&autoplay=1`
      iframeRef.current.src = newSrc
    }
  }, [currentTime, effectiveVideoId])

  const clearInput = () => {
    setVideoLink('')
    setShowPreview(false)
   
    setError('')
    setCurrentTime(0)
  }

  const clearSearch = () => {
    setUserInput('')
    setSearchResults([])
    setError('')
  }

  const jumpToTimestamp = useCallback((timestamp: number,newVideoId?:string) => {
    if(newVideoId)
    setpreviewVideoId(newVideoId)
 
    setCurrentTime(timestamp)
    if (!showPreview) {
      setShowPreview(true)
    }
  },[showPreview])

  const openInYouTube = (url: string) => {
    if (keepExtensionOpen) {
      
      if (typeof chrome !== 'undefined' && chrome.tabs) {
        chrome.tabs.create({ url: url, active: false })
      } else {
        const newWindow = window.open(url, '_blank')
        if (newWindow) {
          newWindow.focus()
        }
      }
    } else {
     
      if (typeof chrome !== 'undefined' && chrome.tabs) {
        chrome.tabs.create({ url: url })
      } else {
        window.open(url, '_blank')
      }
    }
  }

  const youtubesearch =useCallback( async () => {
    if (!userInput.trim()) {
      setError('Please enter a search query')
      return
    }

    setIsLoading(true)
    setError('')
    
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: "POST",
        headers: {
          'Content-type': 'application/json'
        },
        body: JSON.stringify({
          youtube_url: videoLink,
          query: userInput,
          suggestions: enableSuggestions,
          top_k: 5,
          max_related_videos: 4
        })
      })

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`)
      }

      const data:ApiResponseWithSuggestions|ApiResponseWithoutSuggestions= await response.json()
      console.log('data',data);
      setSearchResults(data.input_video.segments || [])
      if(data.suggestions_enabled){
        if(data.related_videos.length<=0){
          setError('No suggestions available for your query')

        }
        const sortedSuggestions=data.related_videos.sort(
          (a,b)=>b.best_segment.score-a.best_segment.score
        )
        setSearchSuggestions(sortedSuggestions)
      }
      
      if (data.input_video.segments && data.input_video.segments.length === 0) {
        setError('No results found for your query')
      }
    } catch (error) {
      if (error instanceof Error) {
        setError(`Error searching video: ${error.message}`)
      } else {
        setError('An unexpected error occurred')
      }
    } finally {
      setIsLoading(false)
    }
  },[userInput,videoLink,enableSuggestions])

  return (
    <div className="w-96 min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4 font-sans">
      {/* Settings Section */}
      <div className="mb-4 p-3 bg-white rounded-xl shadow-sm border border-gray-100">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-gray-700">Auto-Extract from Tab</h3>
            <p className="text-xs text-gray-500">Automatically detect YouTube URLs from current tab</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={autoExtract}
              onChange={(e) => setAutoExtract(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-500 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-red-500"></div>
          </label>
        </div>
        
        {/* Keep Extension Open Toggle */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-100">
          <div>
            <h3 className="text-sm font-semibold text-gray-700">Keep Extension Open</h3>
            <p className="text-xs text-gray-500">Prevent extension from closing when opening links</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={keepExtensionOpen}
              onChange={(e) => setKeepExtensionOpen(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-500 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500"></div>
          </label>
        </div>

        {currentTabUrl && (
          <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
            <span className="text-gray-600">Current tab: </span>
            <span className="font-mono text-gray-800 break-all">{currentTabUrl.slice(0, 50)}...</span>
          </div>
        )}
      </div>

      
      <div className="text-center mb-6">
        <div className="flex items-center justify-center gap-2 mb-2">
          <div className="w-8 h-8 bg-red-500 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M23.498 6.186a2.999 2.999 0 0 0-2.112-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.386.505A2.999 2.999 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a2.999 2.999 0 0 0 2.112 2.136C4.495 20.455 12 20.455 12 20.455s7.505 0 9.386-.505a2.999 2.999 0 0 0 2.112-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
            </svg>
          </div>
          <h1 className="text-xl font-bold text-gray-800">YouTube Timestamp Search</h1>
        </div>
        <p className="text-sm text-gray-600">Search within YouTube videos and jump to exact moments</p>
      </div>

      {/* Input Section */}
      <div className="mb-6 space-y-4">
        <div className="relative">
          <input
            type="text"
            placeholder="Enter YouTube link here..."
            value={videoLink}
            onChange={(e) => setVideoLink(e.target.value)}
            className="w-full px-4 py-3 pr-10 border-2 border-gray-200 rounded-xl focus:border-red-400 focus:outline-none transition-colors text-sm bg-white shadow-sm"
          />
          {videoLink && (
            <button
              onClick={clearInput}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
              title="Clear input"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          )}
        </div>

        <div className="relative">
          <input
            type="text"
            placeholder="Enter your search query here..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && youtubesearch()}
            className="w-full px-4 py-3 pr-10 border-2 border-gray-200 rounded-xl focus:border-red-400 focus:outline-none transition-colors text-sm bg-white shadow-sm"
          />
          {userInput && (
            <button
              onClick={clearSearch}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
              title="Clear search"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          )}
        </div>
        
        {/* Status Indicator */}
        <div className="flex items-center gap-2">
          {videoLink && (
            <div className={`flex items-center gap-1 text-xs px-2 py-1 rounded-full ${
              isValidLink 
                ? 'bg-green-100 text-green-700' 
                : 'bg-red-100 text-red-700'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isValidLink ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              {isValidLink ? 'Valid YouTube Link' : 'Invalid YouTube Link'}
            </div>
          )}
          {keepExtensionOpen && (
            <div className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-700">
              <div className="w-2 h-2 rounded-full bg-blue-500"></div>
              Stay Open Mode
            </div>
          )}
        </div>

         <div className="flex items-center justify-between pt-3 border-t border-gray-100">
  <div>
    <h3 className="text-sm font-semibold text-gray-700">Enable Suggestions</h3>
  </div>
  <label className="relative inline-flex items-center cursor-pointer">
    <input
      type="checkbox"
      checked={enableSuggestions}
      onChange={(e) => setenableuggestions(e.target.checked)}
      className="sr-only peer"
    />
    <div className="w-11 h-6 bg-gray-500 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500">
    
    </div>
  </label>
</div>



        <button
  onClick={youtubesearch}
  disabled={!isValidLink || !userInput.trim() || isLoading}
  className={`w-full py-3 px-4 rounded-xl transition-all duration-200 flex items-center justify-center gap-2 font-medium shadow-md
    ${isLoading ? "bg-red-400 cursor-wait" : "bg-red-500 hover:bg-red-600"}
    disabled:bg-gray-300 disabled:text-gray-600 disabled:cursor-not-allowed text-white`}
>
  {isLoading ? (
    <>
      <span className="relative flex h-4 w-4">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
        <span className="relative inline-flex rounded-full h-4 w-4 bg-white"></span>
      </span>
      <span className="text-sm">Searching...</span>
    </>
  ) : (
    <>
      <svg
        className="w-4 h-4"
        fill="currentColor"
        viewBox="0 0 20 20"
      >
        <path
          fillRule="evenodd"
          d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
          clipRule="evenodd"
        />
      </svg>
      <span className="text-sm">Search Video</span>
    </>
  )}
</button>

      </div>

     
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-xl">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span className="text-sm text-red-700">{error}</span>
          </div>
        </div>
      )}
       {searchResults.length > 0 && (
         <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-5">
       <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
            <path d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" />
          </svg>
          <h2 className="text-lg font-bold text-gray-800">Search Results</h2>
        </div>
        <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded-full shadow-inner">
          {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
        </span>
      </div>
<div className='space-y-3'>
     <div className='space-y-3 max-h-64 overflow-y-auto pr-2'>
     
     
          
          <div className=" pr-2 space-y-2">
            {searchResults.map((result, index) => (
              <div
                key={index}
                className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between mb-2">
                  <button
                    onClick={() => jumpToTimestamp(result.timestamp)}
                    className="flex items-center gap-2 text-red-600 hover:text-red-700 font-medium transition-colors group"
                  >
                    <svg className="w-4 h-4 group-hover:scale-110 transition-transform" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                    </svg>
                    {formatTime(result.timestamp)}
                  </button>
                  <div className="flex items-center gap-1">
                    <svg className="w-3 h-3 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    <span className="text-xs text-gray-500">{(result.score * 100).toFixed(0)}%</span>
                  </div>
                </div>
                
                <p className="text-sm text-gray-700 leading-relaxed mb-2">
                  {result.snippet}
                </p>
                
                <div className="flex justify-end">
                 
                  <button
                    onClick={() => openInYouTube(result.url)}
                    className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1"
                  >
                    Open in YouTube
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"/>
                      <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"/>
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
         </div>
         </div>
      )}
     
      {Searchsuggestions.length > 0 && (
        <div className="space-y-5">
        <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
            <path d="M13 7H7v6h6V7z" />
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM4 10a6 6 0 1112 0A6 6 0 014 10z" clipRule="evenodd" />
          </svg>
          <h2 className="text-lg font-bold text-gray-800">
            Suggestions from Top Videos
          </h2>
        </div>
        <span className="text-sm text-gray-600 bg-red-50 px-2 py-1 rounded-full shadow-inner">
          {Searchsuggestions.length} suggestion{Searchsuggestions.length !== 1 ? 's' : ''}
        </span>
      </div>
     <div className='space-y-3 max-h-80 overflow-y-auto pr-2'>
      
   <div className="bg-gradient-to-br from-red-50 via-white to-gray-50 border border-red-100 rounded-xl shadow p-5">
    

    
      {Searchsuggestions.map((result, index) => (
        <div
          key={index}
          className="bg-white rounded-2xl p-5 shadow-md border border-gray-100 hover:shadow-lg transition-all duration-200"
        >
          <div className="flex items-start justify-between mb-2">
            <div>
              <button
                onClick={() =>
                  jumpToTimestamp(result.best_segment.timestamp, result.video_id)
                }
                className="flex items-center gap-2 text-red-600 hover:text-red-700 font-semibold transition"
              >
                <svg
                  className="w-4 h-4 group-hover:scale-110 transition-transform"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                    clipRule="evenodd"
                  />
                </svg>
                {formatTime(result.best_segment.timestamp)}
              </button>
            </div>

            <div className="flex items-center gap-1 text-yellow-600 text-xs">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <span>{(result.best_segment.score * 100).toFixed(0)}%</span>
            </div>
          </div>

          <h3 className="text-md font-semibold text-gray-800 leading-snug mb-1">
            {result.title}
          </h3>
          <p className="text-sm text-gray-500 mb-1">
            <span className="font-medium text-gray-700">Channel:</span> {result.channel_title}
          </p>
          <p className="text-sm text-gray-600 mb-3 line-clamp-3">
            {result.description}
          </p>

          

          <div className="flex justify-end">
            <button
              onClick={() => openInYouTube(result.url)}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
            >
              Open in YouTube
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
              </svg>
            </button>
          </div>
        </div>
      ))}
    </div>
  </div>
  </div>
)}

      
      

      {/* Preview Toggle */}
      {isValidLink && (
        <div className="mb-4">
          <button
            onClick={() => setShowPreview(!showPreview)}
            className="w-full bg-red-500 hover:bg-red-600 text-white py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-2 shadow-sm font-medium"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
              <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd"/>
            </svg>
            {showPreview ? 'Hide Preview' : 'Show Preview'}
          </button>
        </div>
      )}

      {/* Video Preview */}
      {isValidLink && showPreview && effectiveVideoId && (
        <div className="mb-6">
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <div className="aspect-video bg-gray-100">
              <iframe
                ref={iframeRef}
                src={`https://www.youtube.com/embed/${ effectiveVideoId}${currentTime > 0 ? `?start=${currentTime}&autoplay=1` : ''}`}
                title="YouTube video player"
                className="w-full h-full"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            </div>
            <div className="p-4">
              <p className="text-xs text-gray-600 break-all mb-2">
                Video ID: <span className="font-mono bg-gray-100 px-2 py-1 rounded">{effectiveVideoId}</span>
              </p>
              {currentTime > 0 && (
                <p className="text-xs text-gray-600">
                  Current time: <span className="font-mono bg-blue-100 text-blue-800 px-2 py-1 rounded">{formatTime(currentTime)}</span>
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      {isValidLink && (
        <div className="mb-6">
          <button
            onClick={() => openInYouTube(`https://www.youtube.com/watch?v=${effectiveVideoId}${currentTime > 0 ? `&t=${currentTime}` : ''}`)}
            className="w-full bg-white hover:bg-gray-50 border border-gray-200 text-gray-700 py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-2 shadow-sm font-medium"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"/>
              <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"/>
            </svg>
            Open in YouTube {currentTime > 0 && `(at ${formatTime(currentTime)})`}
          </button>
        </div>
      )}

      {/* Footer */}
      <div className="text-center">
        <p className="text-xs text-gray-500">
          Supports only for youtube.com 
        </p>
      </div>
    </div>
  )
}

export default App