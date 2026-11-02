# Reference tables: provenance and spot check

`ieee_table_iii.csv` and `ieee_table_iv.csv` are byte-for-byte copies of the files the human transcribed from page
images of Myrzashova et al., "Safeguarding Patient Data-Sharing: Blockchain-Enabled Federated Learning in Medical
Diagnostics", IEEE Transactions on Sustainable Computing 10(1), 2025. Their contents have not been edited.

Checked on 2026-10-07 against the PDF (`Safeguarding_Patient_Data-Sharing_Blockchain-Enabled_Federated_Learning_in_
Medical_Diagnostics.pdf`). Both tables are images in the PDF, so the check compared every cell by eye with the
rendered pages:

- Table III ("Comparison of experimental setting in related papers", printed page 179, PDF page 4): every row and
  every cell matches.
- Table IV ("Comparison of experimental results in related papers", printed page 185, PDF page 10): every row and
  every cell matches.

Differences are presentation only, and none changes a value:

- Table III, ref [25]: the paper prints the number of cases as "10 299", with a space as the thousands separator.
  The CSV writes "10,299", like the other rows.
- Table III, BC column: the paper puts a check mark before every consensus name. The CSV column `bc_consensus`
  keeps the names only.
- Both tables: the paper's last row is labelled "Our paper". The CSV labels it "Our paper (Myrzashova et al.)", so
  it cannot be read as ours.
- Table IV: words the paper hyphenates across lines ("Ag-gregation", "Mal-leability") are written whole.

Reference numbers such as [19] or [48] are citation numbers in Myrzashova et al.'s bibliography. They are not
references of ours. Every value in these rows is as reported by the original authors, on their own datasets and
models.
