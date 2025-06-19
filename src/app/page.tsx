
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


type Step = 'folderSelection' | 'documentList' | 'vectorGeneration' | 'qaInterface';

export default function DocumentQueryPage() {
  const [currentStep, setCurrentStep] = useState<Step>('folderSelection');
  const [isLoading, setIsLoading] = useState(false);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const documentContents = simulatedDocuments.map(doc => doc.content);

  const handleLoadSampleDocuments = () => {
    setCurrentStep('documentList');
    toast({
      title: "Sample Documents Loaded",
      description: "Review the list of documents and proceed to generate vectors.",
      className: "bg-secondary border-google-green",
    });
  };

  const handleProcessDocuments = () => {
    setCurrentStep('vectorGeneration');
    setIsLoading(true);
    // Simulate vector generation
    setTimeout(() => {
      setIsLoading(false);
      setCurrentStep('qaInterface');
      toast({
        title: "Documents Processed",
        description: "Vector embeddings generated successfully. You can now ask questions.",
        className: "border-google-green bg-secondary",
        action: <CheckCircle2 className="text-green-500" />,
      });
    }, 2500);
  };

  const handleAskQuestion = async () => {
    if (!question.trim()) {
      toast({ title: "Validation Error", description: "Please enter a question.", variant: "destructive" });
      return;
    }
    setIsLoading(true);
    setAnswer('');
    setError(null);
    try {
      const result = await generateAnswer({ question, documents: documentContents });
      setAnswer(result.answer);
      toast({
        title: "Answer Generated",
        description: "The AI has provided an answer to your question.",
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
                Begin by loading the sample documents to simulate folder selection.
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p className="mb-6 text-muted-foreground">
                This application uses a predefined set of sample documents for demonstration purposes.
              </p>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Button onClick={handleLoadSampleDocuments} size="lg">
                <ListChecks className="mr-2 h-5 w-5" />
                Load Sample Documents
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
                The following sample documents will be processed.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 list-disc list-inside bg-muted p-4 rounded-md">
                {simulatedDocuments.map(doc => (
                  <li key={doc.id} className="text-sm">
                    <FileText className="inline-block mr-2 h-4 w-4 text-muted-foreground" />
                    {doc.name}
                  </li>
                ))}
              </ul>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Button onClick={handleProcessDocuments} size="lg">
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
                <Button onClick={handleAskQuestion} disabled={isLoading} className="w-full" size="lg">
                  {isLoading ? (
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="mr-2 h-5 w-5" />
                  )}
                  Get Answer
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
                    <div className="prose prose-sm max-w-none dark:prose-invert p-4 bg-muted/50 rounded-md whitespace-pre-wrap">
                      {answer}
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
        <Button variant="outline" className="flex items-center justify-start gap-2 py-6 text-base">
          <RefreshCw className="h-5 w-5 text-blue-500" />
          <span>Refresh Vectors</span>
        </Button>
        <Button variant="outline" className="flex items-center justify-start gap-2 py-6 text-base">
          <Trash2 className="h-5 w-5 text-red-500" />
          <span>Delete Vectors</span>
        </Button>
        <Button variant="outline" className="flex items-center justify-start gap-2 py-6 text-base">
          <FileUp className="h-5 w-5 text-green-500" />
          <span>Re-load Files</span>
        </Button>
        <Button variant="outline" className="flex items-center justify-start gap-2 py-6 text-base">
          <Power className="h-5 w-5 text-yellow-500" />
          <span>Restart App</span>
        </Button>
      </CardContent>
    </Card>
  );


  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <AppHeader />
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
      </main>
      <AppFooter />
    </div>
  );
}
