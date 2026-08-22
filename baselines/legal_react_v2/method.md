Directory files description 

This directory contains the following files: 

Crime_sentencing_manual : this is the main manual that shows how to obtain the sentencing for any crime.  

United States code title 18: this is the huge book of references – laws that the manual refers to frequently.  

Case-example-{n} : these are case examples of sentencing memorandum that I found on https://www.courtlistener.com/recap/ 

In each I have made some comments through which you can see some case facts and the ground truth outputs.  

I also showed in comment exactly how the lawyers converted the case facts into sentencing.  

High level view of how one does the sentencing task 

1. Identify the Statutory Count of Conviction 

Identify the specific section of the United States Code (e.g., 18 U.S.C. § 1546) for which the defendant was convicted. This provides the legal foundation for the entire calculation. 

2. Map the Statute to the Guideline (Appendix A) 

Utilize the Statutory Index (Appendix A) in the USSG Manual to "join" the U.S. Code violation to the appropriate Chapter Two Guideline Section (e.g., §2L2.2). This identifies the specific "Rule" that governs the crime type. 

3. Determine Chapter Two Offense Level 

Base Offense Level: Establish the mandatory starting point (integer) provided by the specific guideline. 

Specific Offense Characteristics: Apply numerical enhancements (pluses) or reductions (minuses) based on the specific facts of the crime (e.g., the amount of money stolen or the number of documents involved). 

4. Apply Chapter Three Adjustments 

Apply cross-cutting adjustments that are not crime-specific. These include: 

Victim-Related Adjustments: (e.g., vulnerable victims). 

Role in the Offense: (e.g., was the defendant a leader or a minor participant?). 

Acceptance of Responsibility: A reduction (usually -2 or -3) for defendants who plead guilty and demonstrate remorse. 

