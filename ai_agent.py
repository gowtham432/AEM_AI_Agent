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
CHROMA_DB_DIR = "rag_chroma_db_aem"

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
    Retrieve context using MULTIPLE targeted queries to ensure we get
    complete examples from your trained knowledge base
    """
    try:
        all_retrieved = {
            "dialog_context": "",
            "sling_context": "",
            "htl_context": "",
            "js_context": "",
            "fields_context": ""
        }
        
        # Query 1: Get dialog structure examples
        dialog_query = f"dialog XML structure example granite ui form fields tabs {' '.join([f['type'] for f in fields])}"
        dialog_results = collection.query(query_texts=[dialog_query], n_results=TOP_K)
        dialog_docs = dialog_results.get("documents", [[]])[0]
        all_retrieved["dialog_context"] = "\n\n".join(dialog_docs) if dialog_docs else ""
        
        # Query 2: Get Sling Model examples
        sling_query = f"Sling Model Java @Model @ValueMapValue annotations example {' '.join([f['type'] for f in fields])}"
        sling_results = collection.query(query_texts=[sling_query], n_results=TOP_K)
        sling_docs = sling_results.get("documents", [[]])[0]
        all_retrieved["sling_context"] = "\n\n".join(sling_docs) if sling_docs else ""
        
        # Query 3: Get HTL examples
        htl_query = f"HTL template data-sly-use Sling Model example {' '.join([f['type'] for f in fields])}"
        htl_results = collection.query(query_texts=[htl_query], n_results=TOP_K)
        htl_docs = htl_results.get("documents", [[]])[0]
        all_retrieved["htl_context"] = "\n\n".join(htl_docs) if htl_docs else ""
        
        # Query 4: Get field-specific examples
        field_types_str = " ".join([f['type'] for f in fields])
        fields_query = f"field types examples {field_types_str} sling:resourceType granite properties"
        fields_results = collection.query(query_texts=[fields_query], n_results=TOP_K)
        fields_docs = fields_results.get("documents", [[]])[0]
        all_retrieved["fields_context"] = "\n\n".join(fields_docs) if fields_docs else ""
        
        # Query 5: JS validation if needed (based on context or field types)
        multifield_types = ["Multifield", "Tags picker", "Drop down Field"]
        needs_js = any(f['type'] in multifield_types for f in fields) or "validation" in user_context.lower()
        if needs_js:
            js_query = f"JavaScript validation example clientlib {field_types_str}"
            js_results = collection.query(query_texts=[js_query], n_results=5)
            js_docs = js_results.get("documents", [[]])[0]
            all_retrieved["js_context"] = "\n\n".join(js_docs) if js_docs else ""
        
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
    Generate code by STRICTLY following the patterns from your trained knowledge base
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
    
    # Retrieve ALL relevant context from YOUR knowledge base
    print("🔍 Retrieving context from trained knowledge base...")
    rag_context = retrieve_targeted_context(fields, user_context)
    
    print(rag_context)
    # Build the prompt that FORCES adherence to your examples
    full_prompt = f"""You are an AEM code generator. Your ONLY job is to generate code that EXACTLY matches the patterns and structure from the provided knowledge base examples below.

**CRITICAL INSTRUCTIONS:**
1. The examples below are from a TRAINED knowledge base - they represent the EXACT patterns, structure, depth, and properties that MUST be used
2. Do NOT use generic AEM patterns from your training - ONLY use what's shown in the knowledge base examples
3. Study the structure, property names, nesting levels, and patterns in the examples
4. Your output must match the depth and detail of the examples provided
5. Do NOT simplify or abbreviate - if the examples are detailed, your output must be equally detailed

**FIELDS TO IMPLEMENT:**
{fields_json}

**USER REQUIREMENTS:**
{user_context if user_context else "No additional requirements."}

---

**KNOWLEDGE BASE - DIALOG STRUCTURE (YOUR TRAINED PATTERN):**
{rag_context['dialog_context'] if rag_context['dialog_context'] else "No dialog examples found - use standard Granite UI structure"}

**KNOWLEDGE BASE - FIELD TYPES (YOUR TRAINED PATTERNS):**
{rag_context['fields_context'] if rag_context['fields_context'] else "No field examples found"}

**KNOWLEDGE BASE - SLING MODEL (YOUR TRAINED PATTERN):**
{rag_context['sling_context'] if rag_context['sling_context'] else "No sling model examples found"}

**KNOWLEDGE BASE - HTL (YOUR TRAINED PATTERN):**
{rag_context['htl_context'] if rag_context['htl_context'] else "No HTL examples found"}

**KNOWLEDGE BASE - JS VALIDATION (YOUR TRAINED PATTERN):**
{rag_context['js_context'] if rag_context['js_context'] else "No JS validation examples found - only include if explicitly needed"}

---

**GENERATION RULES:**
1. **Dialog XML:**
   - Follow the EXACT structure from the dialog examples above
   - Use the same nesting levels, tabs, and container structure
   - Copy property names and attributes from the examples (jcr:primaryType, sling:resourceType, etc.)
   - Match the depth and detail level of the examples
   - Replace only the field-specific parts (name, fieldLabel) with the fields from the requirements
   - Keep ALL other properties and structure from the examples

2. **Sling Model Java:**
   - Follow the EXACT class structure from the Sling Model examples above
   - Use the same annotations pattern (@Model, @ValueMapValue, @Default)
   - Keep the same imports and package structure style
   - Match the getter method patterns from the examples
   - Use the exact field names from the requirements
   - Don't add extra methods from returned from the RAG. Just use {fields_json} and {user_context}

3. **HTL:**
   - Follow the EXACT HTL pattern from the examples above
   - Use the same data-sly-use pattern
   - Match the property access pattern shown in examples
   - Keep the same structure for conditionals and loops if shown
   - Only change the actual property names to match requirements

4. **JS Validation:**
   - Only include if examples show it OR if user explicitly requests it
   - Follow the exact pattern from the JS examples
   - Match the validation structure shown

Wrap the output of the dialog using structure in {dialog_template}. You can see the main authoring fields in that, where you can chanage based on prompt and fields 
selected. Like, If I add text field, don't just give me the xml for text field, instead wrap that around template and make sure I get the entire dialog such that
I can use directly without any modifications. In the same way If some context gets to you as add some fields in other tabs, you also have that in my template. Don't
change much.

Don't take node name whatever added in knowledge base. For example, I am giving input as number field and name as number and Name as select a number, make sure
you add appropriate node name. The knowledge base is just has examples. 

For Sling models, add imports and annotations needed accordingly. Take {sling_mappings} as an example, and make sure you are adding the fields from {fields_json}
with making appropriate changes, for example we need to use variable names from dialog and add them with @Valuemapvalue annotation, if I have name in dialog as field 
then my sling model should have the same name, that's how AEM works. Don't confuse between {user_context} and {fields_json} as there may be some similar things in both 
of them. Make sure you differentiate between them. Check what user is asking exactly. If the context is vague think as strong AEM developer and do things. 
Don't overly rely on RAG output. Based on the input change accordingly. When responding sling model please take {fields_json} as your main thing and also {user_context}
and generate. Don't give me getters or setters at all. If you see anything more, please remove it. For example, @Model annotation has all the properties, don't change 
them at all. Based on the context change if added in the prompt. For example if anyone asks how to read Content Fragment, just give them that block of code even if 
you get more from RAG. Be careful with multifield items, I have given just an example how mutlifield works, you have to shift according to {user_context} and 
{fields_json}. For example if someone asks to add number field in multifield, just pull the number field from JCR and add to model. Just take it as an example.
If RAG gives exactly the same, please change according to {user_context} and {fields_json}

For HTL files, take example as {htl_snippets}, In AEM to access JCR we definetly need data-sly-use that bind to our sling model. And we don't want any class or id
or anything added to its attributes. We only need to able to show template with the model and printing all the values added in the dialog output based on {user_context} 
and {fields_json}. When generating HTL, make sure you are using the variables added in dialog. For example I added name in dialog to store in JCR as firstName, that has
to be used in both Java and HTL. Just add modelName defined in data-sly-use with variable name. Don't add anything more or less.

Think yourself as the best AEM developer. Don't think about opensource at all and add unnecessary things to Dialog, HTL, JS or Java.
RAG context is just examples from which we can draw our code based on {user_context} and {fields_json}

Don't return comments. Strictly remove all comments in all files. Just give me code. Comments are for your understanding.
**OUTPUT FORMAT:**
Return ONLY valid JSON:
{{
  "dialog": "<exact dialog XML matching your trained pattern>",
  "sling_model": "<exact Sling Model matching your trained pattern>",
  "htl": "<exact HTL matching your trained pattern>",
  "js_validation": "<JS validation if applicable, otherwise empty string>"
}}
"""

    print("🤖 Generating code with strict adherence to trained patterns...")
    print(f"📊 Context retrieved - Dialog: {len(rag_context['dialog_context'])} chars, Sling: {len(rag_context['sling_context'])} chars, HTL: {len(rag_context['htl_context'])} chars")
    
    try:
        # Call OpenAI API with lower temperature for consistency
        response = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.2,  # Very low temperature to stick to patterns
            messages=[
                {
                    "role": "system", 
                    "content": "You are an AEM code generator that STRICTLY follows provided examples. You replicate patterns exactly as shown in the knowledge base without deviation or simplification."
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
        
        # Basic validation
        if not dialog:
            return ("❌ Failed to generate dialog", "", "", "")
        if not sling_model:
            return ("❌ Failed to generate Sling Model", dialog, "", "")
        if not htl:
            return ("❌ Failed to generate HTL", dialog, sling_model, "")
        
        print("✅ Code generated successfully")
        return (dialog, sling_model, htl, js)
        
    except json.JSONDecodeError as e:
        return (f"❌ JSON parsing error: {str(e)}\n\nRaw output:\n{ai_output[:500]}", "", "", "")
    except Exception as e:
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