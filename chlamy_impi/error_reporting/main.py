from chlamy_impi.paths import get_database_output_dir
from pathlib import Path

def main():
    """We load the error description txt files, and write a new markdown file summarising all these errors together
    """
    f1 = "well_segmentation_errors.txt"
    f2 = "frame_correction_errors.txt"
    f3 = "database_assembly_errors.txt"
    outdir = get_database_output_dir()
    
    # Create output markdown file path
    output_file = "error_summary.md"
    
    # Function to read file and create table rows
    def create_table_rows(filepath, start_index=1):
        try:
            with open(Path(outdir) / filepath, 'r') as f:
                lines = f.readlines()
            # Create table rows with line numbers and content
            return [(i, line.strip()) for i, line in enumerate(lines, start=start_index)]
        except FileNotFoundError:
            return []

    # Read all files and get their contents
    files_and_contents = [
        (f1, create_table_rows(f1, 1)),
        (f2, create_table_rows(f2, 1 + len(create_table_rows(f1)))),
        (f3, create_table_rows(f3, 1 + len(create_table_rows(f1)) + len(create_table_rows(f2))))
    ]
    
    # Create markdown content
    markdown_content = []
    
    for filename, rows in files_and_contents:
        if rows:  # Only add section if file has content
            # Add header
            markdown_content.append(f"\n# {filename}\n")
            # Add table header
            markdown_content.append("| Number | Description |")
            markdown_content.append("|---------|-------------|")
            # Add table rows
            for number, content in rows:
                markdown_content.append(f"| {number} | {content} |")
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(markdown_content))

if __name__ == "__main__":
    main()

