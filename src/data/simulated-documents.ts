export interface SimulatedDocument {
  id: string;
  name: string;
  content: string;
}

export const simulatedDocuments: SimulatedDocument[] = [
  { 
    id: "doc1", 
    name: "Healthcare_Policy_Overview.txt", 
    content: "This document outlines the company's healthcare policy. Standard coverage includes medical, dental, and vision. The annual deductible for individuals is $1000 and $2000 for families. Preventive care is covered at 100%." 
  },
  { 
    id: "doc2", 
    name: "Employee_Eligibility_Guidelines.txt", 
    content: "Full-time employees working 30 hours or more per week are eligible for benefits. New hires are eligible after a 90-day waiting period. Part-time employees and contractors are not eligible for standard benefits packages." 
  },
  { 
    id: "doc3", 
    name: "Claim_Submission_Process.txt", 
    content: "To submit a claim, please fill out Form HC-001 and attach all relevant receipts. Claims can be submitted online via the employee portal or mailed to PO Box 12345. Processing time is typically 15-30 business days." 
  },
  { 
    id: "doc4", 
    name: "Dental_Plan_Details_PPO.txt", 
    content: "The PPO Dental Plan covers two cleanings per year at 100%. Fillings are covered at 80% after deductible. Orthodontic services require pre-authorization and are covered at 50% up to a lifetime maximum of $2000."
  }
];
