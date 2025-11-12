import gradio as gr
from openai import OpenAI
import os
import numpy as np
import chromadb
import json
from chromadb import Client as ChromaClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800
TOP_K = 10  # Increased to get more context
CHROMA_DB_DIR = "rag_chroma_db_aem_new"

# ---------- STEP 1: Initialize Chroma Client ----------
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBED_MODEL
)

files_map = {
    "dialog_template": "aem_knowledge_base/dialog_template.txt",
    "fields_catalog": "aem_knowledge_base/fields_catalog.txt",
    "sling_examples": "aem_knowledge_base/sling_mappings.txt",
    "htl_snippets": "aem_knowledge_base/htl_snippets.txt",
    "js_validation": "aem_knowledge_base/multifield_js_validation.txt",
}

# ---------- STEP 2: Build or Load Vector Store ----------
def build_or_load_chroma():
    print("📦 Initializing or loading Chroma collection...")
    collection = chroma_client.get_or_create_collection(
        name="aem_rag_store",
        embedding_function=openai_ef
    )
    if collection.count() > 0:
        print("✅ Existing Chroma collection loaded successfully.")
        return collection
    
    print("📚 Building new Chroma collection from local files...")
    for name, path in files_map.items():
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunk_size = 800
        overlap = 100
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": name}],
                ids=[f"{name}_{i}"]
            )
    
    print(f"✅ Vector store built with {collection.count()} documents.")
    return collection

collection = build_or_load_chroma()

try:
    with open('aem_knowledge_base/dialog_template.txt', 'r', encoding='utf-8') as file:
        dialog_template = file.read()
    with open('aem_knowledge_base/sling_mappings.txt', 'r', encoding='utf-8') as file:
        sling_mappings = file.read()
    with open('aem_knowledge_base/htl_snippets.txt', 'r', encoding='utf-8') as file:
        htl_snippets = file.read()
except FileNotFoundError:
    print("Error: The file was not found.")
    dialog_template = ""
    sling_mappings = ""
    htl_snippets = ""
except Exception as e:
    print(f"An error occurred: {e}")
    dialog_template = ""
    sling_mappings = ""
    htl_snippets = ""

def retrieve_targeted_context(fields, user_context=""):
    """
    Retrieve context using targeted queries that focus on structure patterns,
    not specific field implementations
    """
    try:
        all_retrieved = {
            "dialog_context": "",
            "sling_context": "",
            "htl_context": "",
            "js_context": "",
            "fields_context": ""
        }
        
        # Get unique field types
        field_types_set = set(f['type'] for f in fields)
        field_types_str = " ".join(field_types_set)
        has_multifield = "Multifield" in field_types_set
        
        # Query 1: Dialog structure (focus on structure, not specific fields)
        dialog_query = "dialog XML granite ui container tabs items structure"
        dialog_results = collection.query(query_texts=[dialog_query], n_results=5)
        dialog_docs = dialog_results.get("documents", [[]])[0]
        all_retrieved["dialog_context"] = "\n\n".join(dialog_docs) if dialog_docs else ""
        
        # Query 2: Sling Model patterns (annotations and COMPLETE structure)
        sling_query = "Sling Model @Model adaptables DefaultInjectionStrategy @ValueMapValue @Default complete class"
        if has_multifield:
            sling_query += " @ChildResource List composite multifield"
        sling_results = collection.query(query_texts=[sling_query], n_results=8)  # Increased from 5
        sling_docs = sling_results.get("documents", [[]])[0]
        all_retrieved["sling_context"] = "\n\n".join(sling_docs) if sling_docs else ""
        
        # Query 3: HTL binding patterns
        htl_query = "HTL data-sly-use model property access syntax"
        htl_results = collection.query(query_texts=[htl_query], n_results=5)
        htl_docs = htl_results.get("documents", [[]])[0]
        all_retrieved["htl_context"] = "\n\n".join(htl_docs) if htl_docs else ""
        
        # Query 4: Field-specific examples (only for referenced field types)
        if field_types_str:
            fields_query = f"{field_types_str} sling:resourceType granite field properties"
            fields_results = collection.query(query_texts=[fields_query], n_results=8)
            fields_docs = fields_results.get("documents", [[]])[0]
            all_retrieved["fields_context"] = "\n\n".join(fields_docs) if fields_docs else ""
        
        # Query 5: JS validation (only if explicitly needed)
        needs_validation = (
            "validation" in user_context.lower() or 
            "validate" in user_context.lower() or
            "required" in user_context.lower()
        )
        
        if needs_validation or has_multifield:
            js_query = "JavaScript clientlib validation coral foundation"
            js_results = collection.query(query_texts=[js_query], n_results=3)
            js_docs = js_results.get("documents", [[]])[0]
            all_retrieved["js_context"] = "\n\n".join(js_docs) if js_docs else ""
        
        # Log what was retrieved
        print(f"📚 Retrieved contexts:")
        for key, value in all_retrieved.items():
            print(f"  - {key}: {len(value)} chars")
        
        return all_retrieved
        
    except Exception as e:
        print(f"❌ RAG Error: {str(e)}")
        return {
            "dialog_context": "",
            "sling_context": "",
            "htl_context": "",
            "js_context": "",
            "fields_context": ""
        }

# --- UI Field Options ---
field_types = [
    "RTE Text Field",
    "Drop down Field (Select)",
    "Tags picker",
    "Text Field",
    "Text Area",
    "Password Field",
    "Number Field",
    "Email Field",
    "Date Picker",
    "Color Field",
    "Check Box",
    "Path Field",
    "Multifield"
]

fields_data = []
extra_context = ""


# --- Gradio callbacks ---
def add_field(selected_type, field_name, field_label, current_list):
    if not field_name or not field_label:
        return current_list, "⚠️ Please enter both a field name and label.", fields_data

    fields_data.append({
        "type": selected_type,
        "name": field_name,
        "label": field_label
    })

    new_entry = f"🧩 **{selected_type}** — Label: `{field_label}`, Name: `{field_name}`"
    if current_list == "### 📋 Fields Added\n_(No fields added yet)_":
        updated_list = f"### 📋 Fields Added\n\n{new_entry}"
    else:
        updated_list = current_list + "\n\n" + new_entry

    return updated_list, f"✅ Added {selected_type} field successfully.", fields_data


def reset_fields():
    global fields_data, extra_context
    fields_data = []
    extra_context = ""
    return "### 📋 Fields Added\n_(No fields added yet)_", "", "", "", "", ""


def set_context_chat(prompt):
    global extra_context
    extra_context = prompt
    return f"🧠 Context added successfully:\n> {prompt}"

def generate_sling_model_with_rag(fields, user_context):
    """
    Generate code using knowledge base as reference patterns, not rigid templates
    """
    if not fields:
        return ("⚠️ Please add at least one field before generating.", "", "", "")

    # Build field requirements
    fields_list = []
    for f in fields:
        fields_list.append({
            "type": f['type'],
            "name": f['name'],
            "label": f['label']
        })
    
    fields_json = json.dumps(fields_list, indent=2)
    
    # Retrieve relevant context
    print("🔍 Retrieving context from trained knowledge base...")
    rag_context = retrieve_targeted_context(fields, user_context)
    
    # Build clearer, prioritized prompt
    full_prompt = f"""You are an expert AEM developer generating component code.

**PRIMARY REQUIREMENTS (HIGHEST PRIORITY):**
Fields to implement:
{fields_json}

User requirements:
{user_context if user_context else "No additional requirements - use standard AEM best practices."}

**YOUR TASK:**
Generate complete, working AEM component code that implements EXACTLY the fields specified above. Use the knowledge base examples below as REFERENCE PATTERNS for structure and syntax, but ADAPT them to match the specific fields and requirements provided.

**CRITICAL RULES:**
1. Implement ONLY the fields from the PRIMARY REQUIREMENTS above
2. For Sling Model: REPLICATE the exact annotation pattern, imports, and class structure from "Sling Model Pattern" section - change ONLY the field names/types to match requirements
3. For Dialog/HTL: Use knowledge base examples as structural patterns and adapt to requirements
4. For multifield: Create composite resources based on the SPECIFIC field types requested, not generic examples
5. Dialog node names should be semantic (e.g., "title" for title field, not generic "item1")
6. Sling Model property names MUST match dialog "name" attributes exactly (AEM requirement)
7. HTL should access Sling Model properties using data-sly-use and property access syntax
8. NO comments in any output
9. NO extra methods beyond getters for the specified fields (no setters, no utility methods unless shown in pattern)

---

**KNOWLEDGE BASE EXAMPLES (Use as patterns, not templates):**

Dialog Structure Pattern:
{rag_context['dialog_context'] if rag_context['dialog_context'] else "Use standard Granite UI container/tabs structure"}

Field Type Examples:
{rag_context['fields_context'] if rag_context['fields_context'] else "Use standard Granite UI field types"}

Sling Model Pattern:
{rag_context['sling_context'] if rag_context['sling_context'] else "Use @Model with adaptables=Resource.class and @ValueMapValue"}

HTL Pattern:
{rag_context['htl_context'] if rag_context['htl_context'] else "Use data-sly-use for model binding"}

JS Validation (if needed):
{rag_context['js_context'] if rag_context['js_context'] and ('validation' in user_context.lower() or any(f['type'] == 'Multifield' for f in fields)) else ""}

---

**DIALOG TEMPLATE STRUCTURE:**
{dialog_template}

**SLING MODEL REFERENCE:**
{sling_mappings}

**HTL REFERENCE:**
{htl_snippets}

**GENERATION GUIDELINES:**

Dialog XML:
- Wrap fields in the dialog template structure shown above
- Use semantic node names based on field purpose (e.g., "pageTitle", "description")
- For multifield: Create composite structure with child fields matching the specified types
- Properties: jcr:primaryType, sling:resourceType, fieldLabel, name

Sling Model - FOLLOW KNOWLEDGE BASE PATTERN EXACTLY:
- COPY the exact @Model annotation structure from the Sling Model Pattern above (including ALL parameters like adaptables, defaultInjectionStrategy, resourceType if present)
- COPY the exact import statements from the pattern
- COPY the package structure style from the pattern
- For each field in {fields_json}:
  * Use @ValueMapValue annotation (copy the style from pattern)
  * Add @Default annotation if shown in pattern
  * Match the exact field naming convention from pattern
  * Property name MUST match dialog "name" attribute exactly
- For multifield: Use the EXACT multifield pattern shown in knowledge base (@ChildResource style, List type, etc.)
- Include ONLY getter methods (NO setters, NO extra utility methods)
- Follow the exact method signature style from pattern (public/private, return types, naming)
- If pattern shows @PostConstruct or init methods, include them; otherwise don't
- Match the code formatting and structure (spacing, ordering) from pattern

HTL:
- Use data-sly-use to bind Sling Model
- Access properties as ${{model.propertyName}}
- No styling, no extra HTML structure
- Follow the exact HTL syntax patterns from knowledge base

**IMPORTANT FOR MULTIFIELD:**
If the user specifies a multifield with specific child field types (e.g., "multifield with text and number fields"):
- Create composite structure in dialog with those specific field types
- In Sling Model, COPY the exact multifield handling pattern from your knowledge base (whether it's @ChildResource, List<Resource>, or custom interface)
- DO NOT default to fragment path or generic examples from RAG
- Adapt to the EXACT fields requested

**SLING MODEL GENERATION CHECKLIST:**
Before generating the Sling Model, verify you are:
- ✓ Using the EXACT @Model annotation with ALL parameters from the knowledge base pattern
- ✓ Including the EXACT import statements from the pattern
- ✓ Using the EXACT package structure style from the pattern  
- ✓ Following the EXACT annotation style for each field (@ValueMapValue, @Default, etc.)
- ✓ Matching field naming conventions from the pattern
- ✓ Using ONLY the field names from {fields_json}, not from example code
- ✓ Including only getter methods (no setters unless pattern shows them)

**OUTPUT FORMAT:**
Return valid JSON only:
{{
  "dialog": "<complete dialog XML>",
  "sling_model": "<complete Java Sling Model>",
  "htl": "<complete HTL template>",
  "js_validation": "<JS validation if needed, otherwise empty string>"
}}"""

    print("🤖 Generating code...")
    print(f"📊 Context lengths - Dialog: {len(rag_context['dialog_context'])} chars, Sling: {len(rag_context['sling_context'])} chars")
    
    try:
        # Use higher temperature for better adaptation while maintaining structure
        response = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.4,  # Balanced: structured but adaptable
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert AEM developer. Generate code that implements user requirements exactly, using knowledge base examples as structural patterns. Prioritize user requirements over example patterns when they conflict."
                },
                {
                    "role": "user", 
                    "content": full_prompt
                },
            ],
            response_format={"type": "json_object"}
        )
        
        ai_output = response.choices[0].message.content
        
        # Parse JSON response
        parsed = json.loads(ai_output)

        dialog = parsed.get("dialog", "").strip()
        sling_model = parsed.get("sling_model", "").strip()
        htl = parsed.get("htl", "").strip()
        js = parsed.get("js_validation", "").strip()
        
        # Validation
        if not dialog:
            return ("❌ Failed to generate dialog", "", "", "")
        if not sling_model:
            return ("❌ Failed to generate Sling Model", dialog, "", "")
        if not htl:
            return ("❌ Failed to generate HTL", dialog, sling_model, "")
        
        print("✅ Code generated successfully")
        return (dialog, sling_model, htl, js)
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {str(e)}")
        print(f"Raw output preview: {ai_output[:500]}")
        return (f"❌ JSON parsing error: {str(e)}\n\nRaw output:\n{ai_output[:500]}", "", "", "")
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return (f"❌ Error generating code: {str(e)}", "", "", "")
   
# --- Gradio Interface ---
with gr.Blocks(theme="soft", title="AEM Component Builder") as demo:
    gr.Markdown("# 🧩 AEM Component Builder")

    with gr.Row():
        with gr.Column(scale=2):
            field_type = gr.Dropdown(
                label="🎯 Select Field Type", 
                choices=field_types, 
                value="Text Field"
            )
        with gr.Column(scale=2):
            field_name = gr.Textbox(
                label="🏷️ Field Name (camelCase)", 
                placeholder="e.g., pageTitle"
            )
        with gr.Column(scale=2):
            field_label = gr.Textbox(
                label="📝 Field Label (Display)", 
                placeholder="e.g., Page Title"
            )

    with gr.Row():
        add_btn = gr.Button("➕ Add Field", variant="primary")
        reset_btn = gr.Button("🔄 Reset All", variant="secondary")

    field_list = gr.Markdown("### 📋 Fields Added\n_(No fields added yet)_")
    status = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("## 💬 Additional Context (Optional)")
    
    context_input = gr.Textbox(
        label="Requirements & Context", 
        placeholder="e.g., Make title required, add validation, use specific tab names, etc.",
        lines=3
    )
    context_status = gr.Markdown("")
    context_input.submit(set_context_chat, inputs=[context_input], outputs=[context_status])

    gr.Markdown("---")
    generate_btn = gr.Button("🚀 Generate AEM Component Code", variant="primary", size="lg")

    gr.Markdown("### 📦 Generated Component Code")
    
    with gr.Tab("🧩 Dialog XML"):
        dialog_output = gr.Code(label="Dialog Configuration", language="html", lines=25)
    
    with gr.Tab("☕ Sling Model"):
        sling_output = gr.Code(label="Java Sling Model", lines=25)
    
    with gr.Tab("🧱 HTL Template"):
        htl_output = gr.Code(label="HTL Template", language="html", lines=20)
    
    with gr.Tab("🧮 JS Validation"):
        js_output = gr.Code(label="JavaScript Validation", language="javascript", lines=20)

    # Event handlers
    generate_btn.click(
        fn=lambda user_ctx: generate_sling_model_with_rag(fields_data, user_ctx),
        inputs=[context_input],
        outputs=[dialog_output, sling_output, htl_output, js_output],
    )

    reset_btn.click(
        fn=reset_fields,
        inputs=[],
        outputs=[field_list, status, dialog_output, sling_output, htl_output, js_output],
    )

    add_btn.click(
        fn=add_field,
        inputs=[field_type, field_name, field_label, field_list],
        outputs=[field_list, status, gr.State(fields_data)],
    )


demo.launch(share=False)