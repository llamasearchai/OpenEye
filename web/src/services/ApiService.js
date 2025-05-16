/**
 * Service for communicating with the OpenVideo API
 */
class ApiService {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  /**
   * Analyze a specific frame from the video
   * @param {number} frameIndex - Index of the frame to analyze
   * @returns {Promise<Object>} Analysis result
   */
  async analyzeFrame(frameIndex) {
    try {
      const response = await fetch(`${this.baseUrl}/analyze_frame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ frame_index: frameIndex }),
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error analyzing frame:', error);
      throw error;
    }
  }
  
  /**
   * Query the video using natural language
   * @param {string} query - Natural language query
   * @returns {Promise<Object>} Query result
   */
  async queryVideo(query) {
    try {
      const response = await fetch(`${this.baseUrl}/query_video`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error querying video:', error);
      throw error;
    }
  }
  
  /**
   * Start the RTSP server
   * @returns {Promise<Object>} RTSP server details
   */
  async startRtspServer() {
    try {
      const response = await fetch(`${this.baseUrl}/start_rtsp_server`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error starting RTSP server:', error);
      throw error;
    }
  }
  
  /**
   * Get metadata about the current video
   * @returns {Promise<Object>} Video metadata
   */
  async getVideoMetadata() {
    try {
      const response = await fetch(`${this.baseUrl}/metadata`);
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error getting video metadata:', error);
      throw error;
    }
  }
  
  /**
   * Open a video source (file, camera, or RTSP URL)
   * @param {string} path - Path to video source
   * @returns {Promise<Object>} Result of opening the video
   */
  async openVideo(path) {
    try {
      const response = await fetch(`${this.baseUrl}/open_video`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ path }),
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error opening video:', error);
      throw error;
    }
  }
}

export default ApiService;