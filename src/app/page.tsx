'use client';

import { useState, useEffect } from 'react';
import { AppHeader } from '@/components/layout/app-header';
import { AppFooter } from '@/components/layout/app-footer';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/hooks/use-toast';
import { generateAnswer } from '@/ai/flows/generate-answer';
import { simulatedDocuments, SimulatedDocument } from '@/data/simulated-documents';
import { Loader2, FileText, AlertTriangle, CheckCircle2, HelpCircle, FolderOpen, ListChecks, MessageSquare, Send, Sparkles, RefreshCw, Trash2, FileUp, Power } from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { TerminalProgressModal } from './components/TerminalProgressModal';
import { CaseIdModal } from './components/CaseIdModal';
import { config, getApiUrl } from '@/config/config';


type Step = 'folderSelection' | 'documentList' | 'vectorGeneration' | 'qaInterface';

export default function DocumentQueryPage() {
  const [currentStep, setCurrentStep] = useState<Step>('folderSelection');
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshingVectors, setIsRefreshingVectors] = useState(false);
  const [isDeletingVectors, setIsDeletingVectors] = useState(false);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const documentContents = simulatedDocuments.map(doc => doc.content);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [allFiles, setAllFiles] = useState<File[]>([]);
  const [fileHandles, setFileHandles] = useState<any[]>([]); // for future use if needed
  const [progressModalOpen, setProgressModalOpen] = useState(false);
  const [progressItems, setProgressItems] = useState<any[]>([]);
  const [vectorTimeout, setVectorTimeout] = useState<NodeJS.Timeout | null>(null);
  const [vectorsExist, setVectorsExist] = useState(true); // Track if vectors exist for Q&A button state
  const [progressModalTitle, setProgressModalTitle] = useState('Vector Generation Progress');

  // Case ID state
  const [caseId, setCaseId] = useState<string>('');
  const [caseIdModalOpen, setCaseIdModalOpen] = useState(false);

  // Admin button handlers
  const handleRefreshVectors = async () => {
    if (!caseId) {
      toast({
        title: 'Case ID Required',
        description: 'Please load documents with a Case ID first.',
        variant: 'destructive',
      });
      return;
    }

    setIsRefreshingVectors(true);
    setProgressModalTitle('Vector Refresh Progress');
    setProgressModalOpen(true);
    
    try {
      // Step 1: Show "Scanning for files..." 
      setProgressItems([{
        file: 'Scanning documents folder...',
        progress: 10,
        status: 'processing',
        message: 'Looking for files to refresh'
      }]);
      
      await new Promise(r => setTimeout(r, 500)); // Brief pause to show scanning
      
      // Step 2: Make API call to get files and start refresh
      const res = await fetch(getApiUrl('refreshAllVectors'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: caseId })
      });
      
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to refresh vectors');
      
      // Step 3: Show files found and simulate processing
      if (data.files_processed && data.files_processed.length > 0) {
        // Initialize with all files as pending
        setProgressItems(data.files_processed.map((file: any) => ({
          file: file.filename,
          progress: 0,
          status: 'pending',
          message: 'Waiting to process...'
        })));
        
        await new Promise(r => setTimeout(r, 300)); // Brief pause to show all files
        
        // Process each file with animation
        for (let i = 0; i < data.files_processed.length; i++) {
          const file = data.files_processed[i];
          
          // Set current file to processing
          setProgressItems(items => items.map((it, idx) => 
            idx === i ? { 
              ...it, 
              status: 'processing', 
              progress: 10,
              message: 'Deleting old vectors...'
            } : it
          ));
          
          await new Promise(r => setTimeout(r, config.files.progressUpdateInterval));
          
          // Update message and progress
          setProgressItems(items => items.map((it, idx) => 
            idx === i ? { 
              ...it, 
              progress: 40,
              message: 'Generating new vectors...'
            } : it
          ));
          
          await new Promise(r => setTimeout(r, config.files.progressUpdateInterval));
          
          // Continue progress
          setProgressItems(items => items.map((it, idx) => 
            idx === i ? { 
              ...it, 
              progress: 70,
              message: 'Building vector database...'
            } : it
          ));
          
          await new Promise(r => setTimeout(r, config.files.progressUpdateInterval));
          
          // Complete
          setProgressItems(items => items.map((it, idx) => 
            idx === i ? { 
              ...it, 
              status: file.success ? 'done' : 'error',
              progress: 100,
              message: file.message || (file.success ? 'Vectors refreshed successfully' : 'Failed to refresh vectors')
            } : it
          ));
          
          await new Promise(r => setTimeout(r, 100)); // Brief pause between files
        }
      } else {
        setProgressItems([{
          file: 'No documents found',
          progress: 100,
          status: 'done',
          message: 'No files to refresh in documents folder'
        }]);
      }
      
      setVectorsExist(true);
      toast({
        title: 'Vectors Refreshed',
        description: data.message,
        className: 'border-google-green bg-secondary',
      });
    } catch (e: any) {
      setProgressItems([{
        file: 'Refresh operation',
        progress: 100,
        status: 'error',
        message: e.message || 'Failed to refresh vectors'
      }]);
      
      toast({
        title: 'Refresh Failed',
        description: e.message || 'Failed to refresh vectors',
        variant: 'destructive',
      });
    } finally {
      setIsRefreshingVectors(false);
      // Hide modal after configured delay
      const timeout = setTimeout(() => setProgressModalOpen(false), config.files.modalAutoCloseDelay);
      setVectorTimeout(timeout);
    }
  };

  const handleDeleteVectors = async () => {
    if (!caseId) {
      toast({
        title: 'Case ID Required',
        description: 'Please load documents with a Case ID first.',
        variant: 'destructive',
      });
      return;
    }

    setIsDeletingVectors(true);
    
    try {
      const res = await fetch(getApiUrl('deleteAllVectors'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: caseId })
      });
      
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to delete vectors');
      
      setVectorsExist(false);
      setAnswer(''); // Clear any existing answer
      
      toast({
        title: 'Vectors Deleted',
        description: data.message,
        className: 'border-yellow-500 bg-secondary',
      });
    } catch (e: any) {
      toast({
        title: 'Delete Failed',
        description: e.message || 'Failed to delete vectors',
        variant: 'destructive',
      });
    } finally {
      setIsDeletingVectors(false);
    }
  };

  const handleRestartApp = () => {
    // Reset all state to initial values
    setCurrentStep('folderSelection');
    setIsLoading(false);
    setIsRefreshingVectors(false);
    setIsDeletingVectors(false);
    setQuestion('');
    setAnswer('');
    setError(null);
    setSelectedFiles([]);
    setAllFiles([]);
    setFileHandles([]);
    setProgressModalOpen(false);
    setProgressItems([]);
    setProgressModalTitle('Vector Generation Progress');
    setVectorsExist(true);
    setCaseId(''); // Reset Case ID
    setCaseIdModalOpen(false);
    
    if (vectorTimeout) {
      clearTimeout(vectorTimeout);
      setVectorTimeout(null);
    }
    
    toast({
      title: 'App Restarted',
      description: 'Application has been reset to initial state.',
      className: 'border-blue-500 bg-secondary',
    });
  };

  const handleLoadSampleDocuments = () => {
    // Open Case ID modal instead of directly proceeding
    setCaseIdModalOpen(true);
  };

  // Handle Case ID submission
  const handleCaseIdSubmit = (submittedCaseId: string) => {
    setCaseId(submittedCaseId);
    setCaseIdModalOpen(false);
    
    // Now trigger folder selection
    setTimeout(() => {
      handleActualFolderSelection();
    }, 100); // Small delay to ensure modal closes first
  };

  const handleCaseIdCancel = () => {
    setCaseIdModalOpen(false);
  };

  // Helper to run Python script for each file
  const runVectorScript = async (files: File[]) => {
    setProgressModalTitle('Vector Generation Progress');
    setProgressModalOpen(true);
    setProgressItems(files.map(f => ({ file: f.name, progress: 0, status: 'pending' })));
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setProgressItems(items => items.map((it, idx) => idx === i ? { ...it, status: 'processing', progress: 10 } : it));
      
      try {
        // Create FormData and upload file to FastAPI
        const formData = new FormData();
        formData.append('file', file);
        formData.append('case_id', caseId);
        
        const res = await fetch(getApiUrl('uploadAndGenerateVectors'), {
          method: 'POST',
          body: formData
        });
        
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Failed to process');
        
        // Check if vectors already existed
        if (data.already_vectorized) {
          // Skip progress simulation for already vectorized files
          setProgressItems(items => items.map((it, idx) => idx === i ? { 
            ...it, 
            status: 'skipped', 
            progress: 100, 
            message: 'Already vectorized' 
          } : it));
        } else {
          // Simulate progress for new vectorization
          for (let p = 20; p <= 100; p += 20) {
            await new Promise(r => setTimeout(r, config.files.progressUpdateInterval));
            setProgressItems(items => items.map((it, idx) => idx === i ? { ...it, progress: p } : it));
          }
          setProgressItems(items => items.map((it, idx) => idx === i ? { 
            ...it, 
            status: 'done', 
            progress: 100,
            message: 'Vectors generated'
          } : it));
        }
      } catch (e: any) {
        setProgressItems(items => items.map((it, idx) => idx === i ? { ...it, status: 'error', progress: 100, message: e.message || 'Error' } : it));
      }
    }
    // Hide modal after configured delay
    const timeout = setTimeout(() => setProgressModalOpen(false), config.files.modalAutoCloseDelay);
    setVectorTimeout(timeout);
  };

  // Updated handleProcessDocuments
  const handleProcessDocuments = async () => {
    setCurrentStep('vectorGeneration');
    setIsLoading(true);
    setProgressModalOpen(true);
    setProgressItems(selectedFiles.map(f => ({ file: f.name, progress: 0, status: 'pending' })));
    await runVectorScript(selectedFiles); // Pass actual File objects, not just names
    setIsLoading(false);
    setCurrentStep('qaInterface');
    setVectorsExist(true); // Vectors now exist after processing
    toast({
      title: 'Documents Processed',
      description: 'Vector embeddings generated successfully. You can now ask questions.',
      className: 'border-google-green bg-secondary',
      action: <CheckCircle2 className="text-green-500" />, 
    });
  };

  const handleAskQuestion = async () => {
    if (!question.trim()) {
      toast({ title: "Validation Error", description: "Please enter a question.", variant: "destructive" });
      return;
    }
    
    if (!caseId) {
      toast({ title: "Case ID Required", description: "Please load documents with a Case ID first.", variant: "destructive" });
      return;
    }
    
    setIsLoading(true);
    setAnswer('');
    setError(null);
    try {
      const res = await fetch(getApiUrl('askQuestion'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: question,
          case_id: caseId
        })
      });
      
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to get answer');
      
      // Format the answer with source information
      let formattedAnswer = data.answer;
      
      if (data.context_found && data.sources && data.sources.length > 0) {
        formattedAnswer += "\n\n📚 **Sources:**\n";
        data.sources.forEach((source: any, index: number) => {
          formattedAnswer += `\n${index + 1}. **${source.document}** (Relevance: ${source.relevance_score})\n`;
          formattedAnswer += `   ${source.preview}\n`;
        });
        formattedAnswer += `\n*${data.message}*`;
      } else {
        formattedAnswer += `\n\n*${data.message}*`;
      }
      
      setAnswer(formattedAnswer);
      toast({
        title: "Answer Generated",
        description: data.context_found ? `Found relevant context from ${data.total_sources} sections` : "Answer from general knowledge",
        className: "border-google-green bg-secondary",
      });
    } catch (e) {
      console.error(e);
      const errorMessage = e instanceof Error ? e.message : "An unknown error occurred while generating the answer.";
      setError(errorMessage);
      toast({ title: "Error Generating Answer", description: errorMessage, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };
  
  const renderStepContent = () => {
    switch (currentStep) {
      case 'folderSelection':
        return (
          <Card className="w-full md:w-4/5 mx-auto shadow-xl border-t-4 border-google-blue">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl font-headline">
                <FolderOpen className="h-7 w-7 text-google-blue" />
                Step 1: Load Documents
              </CardTitle>
              <CardDescription>
                Click below to select a folder containing your documents (PDF, TXT, DOCX).
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p className="mb-6 text-muted-foreground">
                The app will only show PDF, TXT, and DOCX files from the selected folder.
              </p>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Button onClick={handleSelectFolder} size="lg">
                <ListChecks className="mr-2 h-5 w-5" />
                Load Documents
              </Button>
            </CardFooter>
          </Card>
        );
      case 'documentList':
        return (
          <Card className="w-full md:w-4/5 mx-auto shadow-xl border-t-4 border-google-blue">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl font-headline">
                <ListChecks className="h-7 w-7 text-google-blue" />
                Step 1.5: Review Documents
              </CardTitle>
              <CardDescription>
                Select the documents you want to process.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {allFiles.length === 0 ? (
                <div className="text-muted-foreground">No documents found in the selected folder.</div>
              ) : (
                <ul className="space-y-2 bg-muted p-4 rounded-md">
                  {allFiles.map((file) => (
                    <li key={file.name} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={selectedFiles.includes(file)}
                        onChange={() => handleToggleFile(file)}
                        className="accent-google-blue h-4 w-4"
                        id={`file-checkbox-${file.name}`}
                      />
                      <label htmlFor={`file-checkbox-${file.name}`}>{file.name}</label>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
            <CardFooter className="flex justify-center">
              <Button onClick={handleProcessDocuments} size="lg" disabled={selectedFiles.length === 0}>
                <Sparkles className="mr-2 h-5 w-5" />
                Process Documents & Generate Vectors
              </Button>
            </CardFooter>
          </Card>
        );
      case 'vectorGeneration':
        return (
          <Card className="w-full md:w-4/5 mx-auto shadow-xl border-t-4 border-google-yellow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl font-headline">
                <Loader2 className="h-7 w-7 text-google-yellow animate-spin" />
                Step 2: Processing Documents
              </CardTitle>
              <CardDescription>
                Simulating vector generation for the selected documents. This may take a few moments.
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center py-10">
              <Loader2 className="h-12 w-12 text-primary animate-spin mx-auto" />
              <p className="mt-4 text-muted-foreground">Generating embeddings...</p>
            </CardContent>
          </Card>
        );
      case 'qaInterface':
        return (
          <>
            <Card className="w-full md:w-4/5 mx-auto shadow-xl border-t-4 border-google-green">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-2xl font-headline">
                  <HelpCircle className="h-7 w-7 text-google-green" />
                  Step 3: Ask Questions
                </CardTitle>
                <CardDescription>
                  Your documents are ready. Ask any question based on their content.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Type your question here..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={4}
                  className="text-base"
                  aria-label="Question input"
                />
                <Button 
                  onClick={handleAskQuestion} 
                  disabled={isLoading || !vectorsExist || !caseId} 
                  className="w-full" 
                  size="lg"
                >
                  {isLoading ? (
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="mr-2 h-5 w-5" />
                  )}
                  Get Answer {!vectorsExist && '(No Vectors)'} {!caseId && '(No Case ID)'}
                </Button>
              </CardContent>
            </Card>

            {(answer || isLoading || error) && (
              <Card className="w-full md:w-4/5 mx-auto shadow-xl border-t-4 border-google-red mt-8">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-2xl font-headline">
                    <MessageSquare className="h-7 w-7 text-google-red" />
                    Answer
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {isLoading && !answer && (
                    <div className="flex items-center justify-center py-6 text-muted-foreground">
                      <Loader2 className="h-8 w-8 animate-spin mr-3" />
                      Fetching answer...
                    </div>
                  )}
                  {error && (
                    <div className="p-4 bg-destructive/10 text-destructive rounded-md flex items-start gap-3">
                      <AlertTriangle className="h-5 w-5 mt-0.5 shrink-0" />
                      <p className="text-sm">{error}</p>
                    </div>
                  )}
                  {answer && !error && (
                    <div className="space-y-4">
                      <div className="prose prose-sm max-w-none dark:prose-invert p-6 bg-muted/50 rounded-lg border border-border">
                        <div className="whitespace-pre-wrap leading-relaxed text-foreground">
                          {answer.split('\n').map((line, index) => {
                            // Format bold text
                            if (line.includes('**')) {
                              const parts = line.split('**');
                              return (
                                <p key={index} className="mb-2">
                                  {parts.map((part, i) => 
                                    i % 2 === 1 ? <strong key={i} className="font-semibold text-primary">{part}</strong> : part
                                  )}
                                </p>
                              );
                            }
                            // Format source headers
                            if (line.startsWith('📚')) {
                              return <h4 key={index} className="font-semibold text-primary mt-4 mb-2 border-b border-border pb-1">{line}</h4>;
                            }
                            // Format numbered sources
                            if (/^\d+\./.test(line.trim())) {
                              return <p key={index} className="mb-1 font-medium text-secondary-foreground">{line}</p>;
                            }
                            // Format italics
                            if (line.includes('*') && !line.includes('**')) {
                              const italicText = line.replace(/\*(.*?)\*/g, '<em class="text-muted-foreground">$1</em>');
                              return <p key={index} className="mb-2 text-sm" dangerouslySetInnerHTML={{ __html: italicText }} />;
                            }
                            // Regular paragraphs
                            return line.trim() ? <p key={index} className="mb-2">{line}</p> : <br key={index} />;
                          })}
                        </div>
                      </div>
                    </div>
                  )}
                  {!isLoading && !answer && !error && (
                     <p className="text-muted-foreground text-center py-6">Your answer will appear here.</p>
                  )}
                </CardContent>
              </Card>
            )}
          </>
        );
    }
  };

  const AdminConsole = () => (
    <Card className="w-full md:w-4/5 mx-auto shadow-xl mt-12 border-t-4 border-muted-foreground">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl font-headline text-muted-foreground">
          <Sparkles className="h-6 w-6" />
          Admin Console
        </CardTitle>
        <CardDescription>
          Manage document processing and application state.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        <Button 
          variant="outline" 
          className="flex items-center justify-start gap-2 py-6 text-base"
          onClick={handleRefreshVectors}
          disabled={isRefreshingVectors || isDeletingVectors || !caseId}
        >
          <RefreshCw className={`h-5 w-5 text-blue-500 ${isRefreshingVectors ? 'animate-spin' : ''}`} />
          <span>Refresh Vectors</span>
        </Button>
        <Button 
          variant="outline" 
          className="flex items-center justify-start gap-2 py-6 text-base"
          onClick={handleDeleteVectors}
          disabled={isRefreshingVectors || isDeletingVectors || !caseId}
        >
          <Trash2 className="h-5 w-5 text-red-500" />
          <span>Delete Vectors</span>
        </Button>
        <Button 
          variant="outline" 
          className="flex items-center justify-start gap-2 py-6 text-base"
          disabled={true}
        >
          <FileUp className="h-5 w-5 text-gray-400" />
          <span>Re-load Files</span>
        </Button>
        <Button 
          variant="outline" 
          className="flex items-center justify-start gap-2 py-6 text-base"
          onClick={handleRestartApp}
          disabled={isRefreshingVectors || isDeletingVectors}
        >
          <Power className="h-5 w-5 text-yellow-500" />
          <span>Restart App</span>
        </Button>
      </CardContent>
    </Card>
  );


  // Helper to prompt folder selection and read files
  const handleSelectFolder = async () => {
    // First, prompt for Case ID
    setCaseIdModalOpen(true);
  };

  // New function to handle actual folder selection after Case ID is set
  const handleActualFolderSelection = async () => {
    try {
      // @ts-ignore
      const dirHandle = await window.showDirectoryPicker();
      const files: File[] = [];
      const handles: any[] = [];
      for await (const entry of dirHandle.values()) {
        if (entry.kind === 'file') {
          const ext = entry.name.split('.').pop()?.toLowerCase();
          if (['pdf', 'txt', 'docx'].includes(ext || '')) {
            const file = await entry.getFile();
            files.push(file);
            handles.push(entry);
          }
        }
      }
      setAllFiles(files);
      setSelectedFiles(files); // default: all selected
      setFileHandles(handles);
      setCurrentStep('documentList');
      toast({
        title: 'Folder Loaded',
        description: `${files.length} document(s) found for Case ID: ${caseId}`,
        className: 'bg-secondary border-google-green',
      });
    } catch (e) {
      toast({
        title: 'Folder Selection Cancelled',
        description: 'No folder was selected.',
        variant: 'destructive',
      });
    }
  };

  // Checkbox handler
  const handleToggleFile = (file: File) => {
    setSelectedFiles((prev) =>
      prev.includes(file)
        ? prev.filter((f) => f !== file)
        : [...prev, file]
    );
  };

  // Modal close handler
  const handleCloseProgressModal = () => {
    setProgressModalOpen(false);
    if (vectorTimeout) clearTimeout(vectorTimeout);
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <AppHeader />
      
      {/* Case ID Display */}
      {caseId && (
        <div className="border-b bg-muted/30 px-4 py-2">
          <div className="container mx-auto flex justify-end">
            <span className="text-sm font-medium text-muted-foreground">
              Case ID: <span className="text-foreground">{caseId}</span>
            </span>
          </div>
        </div>
      )}
      
      <main className="flex-grow container mx-auto px-4 py-8 sm:py-12">
        <div className="space-y-8 w-full">
          {renderStepContent()}
        </div>
        {currentStep === 'qaInterface' && (
          <>
            <Separator className="my-12" />
            <AdminConsole />
          </>
        )}
        <TerminalProgressModal 
          open={progressModalOpen} 
          onClose={handleCloseProgressModal} 
          items={progressItems} 
          title={progressModalTitle}
        />
        <CaseIdModal
          open={caseIdModalOpen}
          onClose={handleCaseIdCancel}
          onSubmit={handleCaseIdSubmit}
        />
      </main>
      <AppFooter />
    </div>
  );
}
