import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

interface CaseIdModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (caseId: string) => void;
}

export const CaseIdModal: React.FC<CaseIdModalProps> = ({ open, onClose, onSubmit }) => {
  const [caseId, setCaseId] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!caseId.trim()) {
      setError('Case ID is required');
      return;
    }
    
    // Validate case ID format (alphanumeric, dashes, underscores allowed)
    const caseIdRegex = /^[a-zA-Z0-9_-]+$/;
    if (!caseIdRegex.test(caseId.trim())) {
      setError('Case ID can only contain letters, numbers, dashes, and underscores');
      return;
    }
    
    setError('');
    onSubmit(caseId.trim());
    setCaseId('');
  };

  const handleCancel = () => {
    setCaseId('');
    setError('');
    onClose();
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-background rounded-lg shadow-2xl w-full max-w-md border border-border relative">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4 text-foreground">Enter Case ID</h2>
          <p className="text-muted-foreground mb-4 text-sm">
            Please provide a Case ID to organize your documents. This will be used to group all related documents together.
          </p>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="caseId">Case ID</Label>
              <Input
                id="caseId"
                type="text"
                placeholder="e.g., CASE-2025-001, PROJECT-A, etc."
                value={caseId}
                onChange={(e) => setCaseId(e.target.value)}
                className="w-full"
                autoFocus
              />
              {error && (
                <p className="text-destructive text-sm">{error}</p>
              )}
            </div>
            
            <div className="flex gap-2 justify-end pt-4">
              <Button 
                type="button" 
                variant="outline" 
                onClick={handleCancel}
              >
                Cancel
              </Button>
              <Button type="submit">
                Continue
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
