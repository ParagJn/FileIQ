// Configuration file for FileIQ application

export const config = {
  // API Configuration
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8001',
    endpoints: {
      uploadAndGenerateVectors: '/upload-and-generate-vectors',
      generateVectors: '/generate-vectors',
      askQuestion: '/ask-question',
      refreshAllVectors: '/refresh-all-vectors',
      deleteAllVectors: '/delete-all-vectors',
    }
  },
  
  // Server Configuration
  server: {
    backend: {
      host: 'localhost',
      port: 8001
    },
    frontend: {
      host: 'localhost', 
      port: 3000
    }
  },

  // Application Settings
  app: {
    name: 'FileIQ',
    description: 'Multi-step document Q&A web application',
    version: '1.0.0'
  },

  // File Processing Settings
  files: {
    supportedExtensions: ['.pdf', '.docx', '.txt', '.json'] as const,
    maxFileSize: 100 * 1024 * 1024, // 100MB
    progressUpdateInterval: 200, // milliseconds
    modalAutoCloseDelay: 3000 // milliseconds
  },

  // Vector DB Settings
  vectorDB: {
    batchSize: 32,
    cacheSize: 1000,
    scoreThreshold: 0.3,
    maxResults: 3
  },

  // UI Settings
  ui: {
    progressSteps: [20, 40, 60, 80, 100],
    toastDuration: 5000,
    loadingSpinnerDelay: 100
  }
} as const;

// Helper function to get full API URL
export const getApiUrl = (endpoint: keyof typeof config.api.endpoints): string => {
  return `${config.api.baseUrl}${config.api.endpoints[endpoint]}`;
};

// Helper function to check if file type is supported
export const isSupportedFileType = (filename: string): boolean => {
  const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));
  return config.files.supportedExtensions.includes(extension as any);
};

export default config;
