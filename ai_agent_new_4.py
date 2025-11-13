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
TOP_K = 10
CHROMA_DB_DIR = "rag_chroma_db_aem_new_2"

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
            sling_query += " @ChildResource @PostConstruct ValueMap POJO inner class ArrayList multifield composite"
        sling_results = collection.query(query_texts=[sling_query], n_results=8)
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

    new_entry = f"🧩 **{selected_type}** – Label: `{field_label}`, Name: `{field_name}`"
    if current_list == "### 📋 Fields Added\n_(No fields added yet)_":
        updated_list = f"### 📋 Fields Added\n\n{new_entry}"
    else:
        updated_list = current_list + "\n\n" + new_entry

    return updated_list, f"✅ Added {selected_type} field successfully.", fields_data


def reset_fields():
    global fields_data, extra_context
    fields_data = []
    extra_context = ""
    return "### 📋 Fields Added\n_(No fields added yet)_", "", "", "", ""


def set_context_chat(prompt):
    global extra_context
    extra_context = prompt
    return f"🧠 Context added successfully:\n> {prompt}"

def generate_sling_model_with_rag(fields, user_context):
    """
    Generate code using knowledge base as reference patterns, not rigid templates
    """
    if not fields:
        return ("⚠️ Please add at least one field before generating.", "", "")

    # Build field requirements
    fields_list = []
    multifield_info = {}
    
    for f in fields:
        fields_list.append({
            "type": f['type'],
            "name": f['name'],
            "label": f['label']
        })
        # Track multifield for special handling
        if f['type'] == "Multifield":
            multifield_info[f['name']] = {
                "label": f['label'],
                "child_fields": []
            }
    
    fields_json = json.dumps(fields_list, indent=2)
    
    # Detect tab organization from context
    tab_instructions = ""
    if user_context:
        context_lower = user_context.lower()
        
        # Check for any tab-related keywords
        has_tab_mention = any(keyword in context_lower for keyword in [
            'separate tab', 'different tab', 'another tab', 'in a tab',
            'tab with name', 'tab name as', 'tab named',
            'configuration tab', 'content tab', 'properties tab',
            'settings tab', 'data tab', 'additional tab'
        ])
        
        if has_tab_mention:
            # Extract all mentioned tab names from context
            import re
            tab_patterns = [
                r'tab (?:with )?name (?:as |is )?["\']?([A-Za-z\s]+)["\']?',
                r'in (?:a |the )?["\']?([A-Za-z\s]+)["\']? tab',
                r'separate tab (?:with name |as )?["\']?([A-Za-z\s]+)?["\']?',
                r'another tab (?:with name |as )?["\']?([A-Za-z\s]+)?["\']?',
            ]
            
            detected_tabs = []
            for pattern in tab_patterns:
                matches = re.findall(pattern, context_lower, re.IGNORECASE)
                detected_tabs.extend([m.strip().title() for m in matches if m.strip()])
            
            # Remove duplicates while preserving order
            detected_tabs = list(dict.fromkeys(detected_tabs))
            
            tab_analysis = f"Detected tab names from context: {', '.join(detected_tabs) if detected_tabs else 'No explicit names found'}"
            
            tab_instructions = f"""
**TAB ORGANIZATION DETECTED:**
The user context mentions tab organization.

{tab_analysis}

**INSTRUCTIONS FOR GENERATING TABS:**
1. **Analyze the context carefully** to determine:
   - How many tabs are needed (can be 1, 2, 3, or more)
   - What each tab should be named
   - Which fields belong in which tab

2. **Tab naming rules:**
   - Tab node names: lowercase/camelCase (e.g., properties, items, additionalDetails, configuration)
   - Tab titles (jcr:title): Display-friendly, properly capitalized (e.g., "Properties", "Items", "Additional Details", "Configuration")
   - Extract tab names from phrases like:
     * "in [TabName] tab" → use TabName
     * "tab with name as [TabName]" → use TabName
     * "separate tab" with context clues → infer appropriate name
     * "another tab" → infer from fields assigned to it

3. **Field distribution logic:**
   - Parse context for explicit tab assignments (e.g., "Add X in [TabName] tab")
   - If a field is mentioned with a specific tab, put it in that tab
   - If "multifield in separate tab" without name → use "Items" as default
   - If "another tab" for certain fields → infer tab name from field context or use generic name
   - Fields not assigned to any tab → put in first tab (usually "Properties")

4. **CRITICAL TAB STRUCTURE:** Each tab MUST follow this exact nesting:
   ```
   <tabNodeName jcr:primaryType="nt:unstructured" 
                jcr:title="Display Name" 
                sling:resourceType="granite/ui/components/coral/foundation/container" 
                margin="{{Boolean}}true">
       <items jcr:primaryType="nt:unstructured">
           <columns jcr:primaryType="nt:unstructured" 
                    sling:resourceType="granite/ui/components/coral/foundation/fixedcolumns" 
                    margin="{{Boolean}}true">
               <items jcr:primaryType="nt:unstructured">
                   <column jcr:primaryType="nt:unstructured" 
                           sling:resourceType="granite/ui/components/coral/foundation/container">
                       <items jcr:primaryType="nt:unstructured">
                           <!-- YOUR FIELDS GO HERE -->
                       </items>
                   </column>
               </items>
           </columns>
       </items>
   </tabNodeName>
   ```

5. **Examples of tab detection:**
   - "Add multifield in separate tab with name as Items" → Create "Items" tab for multifield
   - "Add email and path in another tab with name as Additional Details" → Create "Additional Details" tab
   - "Put X in Configuration tab, Y in Content tab" → Create both tabs with those fields
   - If context mentions 3+ different tabs → create all of them

6. **DO NOT:**
   - Hardcode to only 2 tabs
   - Put multifield directly as a tab (always wrap in proper tab structure)
   - Skip any tabs mentioned in context
   - Assume default tab names when explicit names are given

**EXAMPLE: 3-Tab Structure**
Context: "Add multifield in Items tab, Add email and path in Additional Details tab"
→ Create 3 tabs:
```
<tabs>
    <items>
        <properties jcr:title="Properties">
            <items><columns><items><column><items>
                <!-- Fields not assigned to specific tabs -->
                <text.../> <dropdown.../> <color.../>
            </items></column></items></columns></items>
        </properties>
        <items jcr:title="Items">
            <items><columns><items><column><items>
                <!-- Multifield -->
                <items (multifield)>...</items>
            </items></column></items></columns></items>
        </items>
        <additionalDetails jcr:title="Additional Details">
            <items><columns><items><column><items>
                <!-- Email and path -->
                <email.../> <path.../>
            </items></column></items></columns></items>
        </additionalDetails>
    </items>
</tabs>
```
"""
    
    # Parse multifield child fields from context if mentioned
    multifield_context_note = ""
    if multifield_info and user_context:
        # Extract component-level field names to avoid conflicts
        component_field_names = [f['name'] for f in fields_list]
        component_field_types = [f['type'] for f in fields_list]
        
        multifield_context_note = f"""
**MULTIFIELD STRUCTURE CLARIFICATION:**
You have a Multifield component named: {', '.join(multifield_info.keys())}
The user context describes what fields go INSIDE this multifield (as child fields).

CRITICAL: Component-level fields vs Multifield child fields:
- Component-level fields from dropdown: {component_field_names}
- Multifield child fields: specified in user context (e.g., "add text, number, path to multifield")

THESE ARE COMPLETELY SEPARATE:
- If "Text Field" is in component fields AND "text field" is mentioned for multifield → Create BOTH
- Component field uses name from dropdown (e.g., "text")
- Multifield child fields use semantic names based on context (e.g., "itemText", "itemNumber", "itemPath")
- If names would conflict, prefix multifield child fields with "item" or use context clues

IMPLEMENTATION REQUIREMENTS:
1. Dialog: 
   - Component-level fields go in appropriate tab (outside multifield)
   - Multifield child fields go INSIDE the multifield composite structure
   - Ensure NO name conflicts between component fields and multifield child fields
   
2. Sling Model: Use the POJO pattern with @PostConstruct initialization
   - @ChildResource Resource container for the multifield
   - private List<PojoClass> items = new ArrayList<>();
   - @PostConstruct method to iterate children and build POJO list
   - Inner static POJO class with fields matching child field names (from multifield context)
   - Use semantic POJO class name based on context (e.g., "FragmentItem", "CardItem", "MenuItem")
   
3. HTL: Access POJO fields directly (e.g., ${{item.fieldName}})

EXAMPLE:
If component has "Text Field (name: text)" AND context says "add text field to multifield":
Dialog should have:
  <text> (component field, name="./text")
  <items> (multifield)
    <field>
      <items>
        <itemText> (multifield child, name="./itemText") 
      </items>
    </field>
  </items>
"""
    
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

{tab_instructions}

{multifield_context_note}

**YOUR TASK:**
Generate complete, working AEM component code that implements EXACTLY the fields specified above. Use the knowledge base examples below as REFERENCE PATTERNS for structure and syntax, but ADAPT them to match the specific fields and requirements provided.

**CRITICAL RULES:**
1. Implement ONLY the fields from the PRIMARY REQUIREMENTS above
2. NEVER skip or omit fields listed in "Fields to implement" - ALL must be included in dialog, Sling Model, and HTL
3. For Sling Model: 
   - REPLICATE the exact annotation pattern, imports, and class structure from "Sling Model Pattern" section
   - Create @ValueMapValue for EVERY component-level field (Text, Dropdown, Color, CheckBox, etc.)
   - Field names MUST match dialog "name" attributes exactly
   - For multifield: ONLY create @ChildResource + POJO pattern (NO @ValueMapValue for child fields)
4. For HTL:
   - Display ALL component-level fields using ${{model.fieldName}}
   - For multifield: iterate POJO list and access child fields via item
   - NEVER confuse component field names with multifield child field names
5. CAREFULLY distinguish between:
   - Component-level fields (listed in "Fields to implement" - MUST ALL be in dialog, Sling Model @ValueMapValue, and HTL)
   - Multifield container (if a Multifield type is listed)
   - Multifield child fields (specified in user context, NOT in component fields list - go INSIDE multifield, in POJO only)
6. If multifield child fields AND similar field types exist in component fields:
   - These are SEPARATE fields that MUST BOTH be included
   - Component field: uses exact name from "Fields to implement", has @ValueMapValue, displayed in HTL
   - Multifield child: uses different name to avoid conflicts (e.g., "itemText" vs "text"), in POJO only, accessed via iteration
7. For Dialog tabs: Parse user context for tab organization instructions
   - Keywords: "separate tab", "in [TabName] tab", "Tab1", "Tab2", etc.
   - Create appropriate tab structure with semantic node names and display titles
   - If no tab instructions: use single "Properties" tab
   - **CRITICAL:** Each tab MUST have full structure: tabNode → items → columns → items → column → items → [fields]
   - NEVER put fields (including multifield) directly under tabs/items - always wrap in proper tab structure
8. Dialog node names should be semantic (e.g., "title" for title field, not generic "item1")
9. NO comments in any output
10. NO extra methods beyond getters for the specified fields (no setters, no utility methods unless shown in pattern)

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
- **INCLUDE ALL FIELDS from "Fields to implement"** - do not skip any component-level fields
- **TAB ORGANIZATION:**
  * **DEFAULT BEHAVIOR:**
    - If context says "multifield in separate tab or add some fields in multifield" OR "separate tab": Create tabs according to number of mentioned in context
      Based on the context add tab names. If nothing mentioned just add the one with data you have trained.
    - If context specifies custom tab names (e.g., "in Configuration tab", "in Content tab"): Use those exact names and distribute fields as mentioned
    - Only create multiple tabs if the user context explicitly mentions separate tabs or specific tab names.
    - Otherwise, include ALL fields (including multifield) inside a single "Properties" tab.
    - If NO tab mention: All fields in one "Properties" tab
  * Tab node names should be lowercase/camelCase (e.g., properties, itemsTab, configuration)
  * Tab titles (jcr:title) should be display-friendly (e.g., "Properties", "Items", "Configuration")
  * **CRITICAL TAB STRUCTURE:** Each tab MUST follow this exact nesting:
    ```
    <tabNodeName jcr:primaryType="nt:unstructured" jcr:title="Display Name" sling:resourceType="granite/ui/components/coral/foundation/container" margin="{{Boolean}}true">
        <items jcr:primaryType="nt:unstructured">
            <columns jcr:primaryType="nt:unstructured" sling:resourceType="granite/ui/components/coral/foundation/fixedcolumns" margin="{{Boolean}}true">
                <items jcr:primaryType="nt:unstructured">
                    <column jcr:primaryType="nt:unstructured" sling:resourceType="granite/ui/components/coral/foundation/container">
                        <items jcr:primaryType="nt:unstructured">
                            <!-- YOUR FIELDS GO HERE -->
                        </items>
                    </column>
                </items>
            </columns>
        </items>
    </tabNodeName>
    ```
  * DO NOT put multifield directly as a tab sibling - it must be wrapped in a proper tab with the structure above
  * Reference the provided Dialog Template for exact tab structure syntax
- **For multifield composite structure:**
  * The multifield itself is a component-level field (from "Fields to implement")
  * Child fields specified in context go INSIDE the multifield composite node
  * If multifield child fields have same TYPE as component fields → use different node names to avoid conflicts
  * Example: Component has "text" field, multifield needs text field → use "itemText" or "fieldText" for multifield child
  * Node naming for multifield children: semantic based on context (itemText, itemNumber, itemPath, etc.)
- Properties: jcr:primaryType, sling:resourceType, fieldLabel, name

**EXAMPLE TAB SCENARIOS:**

Scenario 1: Context says "multifield in separate tab" or just "separate tab"
Component fields: Text Field, Checkbox, Dropdown, Multifield
→ Create 2 tabs:
  - Tab 1 "Properties": Text Field, Checkbox, Dropdown (all non-multifield fields)
  - Tab 2 "Items": Multifield only
```
<tabs>
    <items>
        <properties jcr:title="Properties">
            <items><columns><items><column><items>
                <text.../> <checkbox.../> <dropdown.../>
            </items></column></items></columns></items>
        </properties>
        <itemsTab jcr:title="Items">
            <items><columns><items><column><items>
                <items (multifield)>...</items>
            </items></column></items></columns></items>
        </itemsTab>
    </items>
</tabs>
```

Scenario 2: "Put text and dropdown in Content tab, multifield in Configuration tab"
→ Create 2 tabs with specified names:
```
<tabs>
    <items>
        <content jcr:title="Content">
            <items><columns><items><column><items>
                <text.../> <dropdown.../>
            </items></column></items></columns></items>
        </content>
        <configuration jcr:title="Configuration">
            <items><columns><items><column><items>
                <multifield.../>
            </items></column></items></columns></items>
        </configuration>
    </items>
</tabs>
```

Scenario 3: No tab mention in context
→ Create 1 tab: <properties jcr:title="Properties"> with ALL fields (including multifield) inside proper structure

**CRITICAL:** Every tab needs: tabNode → items → columns → items → column → items → [your fields]
DO NOT skip any of these nested levels!

**EXAMPLE FIELD CONFLICT RESOLUTION:**
Component fields: Text Field (name: text), Check Box (name: qsp), Multifield (name: items), Dropdown (name: color)
Context: "Add text, number, path to multifield"

**Dialog structure:**
```
<text name="./text" fieldLabel="Text Field"> (component field - MUST be included)
<qsp name="./qsp"> (component field - MUST be included)
<items> (multifield container)
  <field name="./items">
    <items>
      <itemText name="./itemText"> (multifield child - different name!)
      <itemNumber name="./itemNumber">
      <itemPath name="./itemPath">
    </items>
  </field>
</items>
<color name="./color"> (component field - MUST be included)
```

**Sling Model:**
```
@ValueMapValue private String text;  // Component field
@ValueMapValue private Boolean qsp;  // Component field
@ValueMapValue private String color; // Component field

@ChildResource(name = "items")
private Resource itemsContainer;     // Multifield container

private List<ItemData> items = new ArrayList<>();  // POJO list

@PostConstruct
protected void init() {{
    // Build POJO list from multifield children
}}

public String getText() {{ return text; }}    // Component getter
public Boolean getQsp() {{ return qsp; }}      // Component getter
public String getColor() {{ return color; }}   // Component getter
public List<ItemData> getItems() {{ return items; }}  // Multifield getter

public static class ItemData {{
    private final String itemText;      // Multifield child fields
    private final Integer itemNumber;
    private final String itemPath;
    // Constructor and getters...
}}
```

**HTL:**
```
<div data-sly-use.model="com.example.Model">
    <p>Text: ${{model.text}}</p>        <!-- Component field -->
    <p>QSP: ${{model.qsp}}</p>          <!-- Component field -->
    <p>Color: ${{model.color}}</p>      <!-- Component field -->
    
    <ul data-sly-list.item="${{model.items}}">  <!-- Multifield iteration -->
        <li>${{item.itemText}} - ${{item.itemNumber}} - ${{item.itemPath}}</li>
    </ul>
</div>
```

Note: ALL component fields (text, qsp, color) appear in all three files. Multifield children (itemText, itemNumber, itemPath) only in POJO and HTL iteration.

Sling Model - FOLLOW KNOWLEDGE BASE PATTERN EXACTLY:
- COPY the exact @Model annotation structure from the Sling Model Pattern above (including ALL parameters like adaptables, defaultInjectionStrategy, resourceType if present)
- COPY the exact import statements from the pattern (including ArrayList, List, ValueMap, etc.)
- COPY the package structure style from the pattern
- **CREATE @ValueMapValue FOR ALL COMPONENT-LEVEL FIELDS** (from "Fields to implement"):
  * Text Field → @ValueMapValue String fieldName; (use exact name from "Fields to implement")
  * Dropdown → @ValueMapValue String fieldName;
  * Color Field → @ValueMapValue String fieldName;
  * Check Box → @ValueMapValue Boolean fieldName;
  * Number Field → @ValueMapValue Integer fieldName;
  * Date Picker → @ValueMapValue Date fieldName;
  * Path Field → @ValueMapValue String fieldName;
  * Tags picker → @ValueMapValue String[] fieldName;
  * RTE Text Field → @ValueMapValue String fieldName;
  * Text Area → @ValueMapValue String fieldName;
  * Password Field → @ValueMapValue String fieldName;
  * Email Field → @ValueMapValue String fieldName;
  * Multifield → Use POJO pattern (see below)
- **For multifield with child fields:**
  * @ChildResource(name = "multifieldName") Resource containerResource; (use exact name from "Fields to implement")
  * private List<PojoClass> multifieldList = new ArrayList<>(); (semantic name for the list)
  * @PostConstruct init() method to populate the list
  * Inner static POJO class with fields matching MULTIFIELD CHILD field names (NOT component field names)
  * POJO field names should match the dialog node names used for multifield children (e.g., itemText, itemNumber, itemPath)
  * Getter returns List<PojoClass> NOT List<Resource>
- **Include getter for EVERY field:**
  * Component-level fields: public Type getFieldName() {{ return fieldName; }}
  * Multifield: public List<PojoClass> getMultifieldList() {{ return multifieldList; }}
- Match the exact method signature style from pattern (public/private, return types, naming)
- If pattern shows ComponentExporter, include it; otherwise use simpler pattern
- Required imports for multifield: ArrayList, List, ValueMap, @PostConstruct, StringUtils or appropriate defaults

**SLING MODEL FIELD MAPPING EXAMPLE:**
If "Fields to implement" has: Text Field (name: text), Multifield (name: items)
And context says: "Add text, number, path to multifield"
Sling Model should have:
```
@ValueMapValue
private String text;  // Component-level field

@ChildResource(name = "items")
private Resource itemsContainer;  // Multifield container

private List<ItemData> items = new ArrayList<>();  // POJO list

@PostConstruct
protected void init() {{
    if (itemsContainer != null) {{
        for (Resource item : itemsContainer.getChildren()) {{
            ValueMap vm = item.getValueMap();
            items.add(new ItemData(
                vm.get("itemText", ""),      // Multifield child field
                vm.get("itemNumber", 0),     // Multifield child field
                vm.get("itemPath", "")       // Multifield child field
            ));
        }}
    }}
}}

public String getText() {{ return text; }}  // Component field getter
public List<ItemData> getItems() {{ return items; }}  // Multifield getter

public static class ItemData {{
    private final String itemText;    // Multifield child fields
    private final Integer itemNumber;
    private final String itemPath;
    // Constructor and getters...
}}
```

HTL:
- Use data-sly-use to bind Sling Model
- Access properties as ${{model.propertyName}}
- For multifield with POJO: iterate with data-sly-list="${{model.items}}" and access fields directly as ${{item.fieldName}}
- Example: <li data-sly-list.item="${{model.items}}">${{item.text}} - ${{item.number}}</li>
- No styling, no extra HTML structure
- Follow the exact HTL syntax patterns from knowledge base

**IMPORTANT FOR MULTIFIELD:**
If a Multifield is in "Fields to implement" AND context specifies child fields:

DIALOG STRUCTURE:
```
<items jcr:primaryType="nt:unstructured"
       sling:resourceType="granite/ui/components/coral/foundation/form/multifield"
       fieldLabel="Add Items">
    <field jcr:primaryType="nt:unstructured"
           sling:resourceType="granite/ui/components/coral/foundation/container"
           name="./items">
        <items jcr:primaryType="nt:unstructured">
            <text .../>  <!-- Child fields go here -->
            <number .../>
            <path .../>
        </items>
    </field>
</items>
```

SLING MODEL:
```
@ChildResource(name = "items")
private List<Resource> items;

public List<Resource> getItems() {{
    return items;
}}
```
NO separate @ValueMapValue fields for text/number/path - those are accessed via items.get(i).getValueMap().get("text")

HTL ACCESS:
```
<ul data-sly-list.item="${{model.items}}">
    <li>${{item.valueMap.text}} - ${{item.valueMap.number}} - ${{item.valueMap.path}}</li>
</ul>
```

**SLING MODEL GENERATION CHECKLIST:**
Before generating the Sling Model, verify you are:
- ✓ Using the EXACT @Model annotation with ALL parameters from the knowledge base pattern
- ✓ Including the EXACT import statements from the pattern
- ✓ Using the EXACT package structure style from the pattern  
- ✓ Following the EXACT annotation style for each field (@ValueMapValue, @Default, etc.)
- ✓ Creating @ValueMapValue fields for EVERY component-level field from "Fields to implement" (count them and verify!)
- ✓ Field names in Sling Model MUST match the "name" attribute from dialog exactly
- ✓ For multifield: Creating POJO inner class + @PostConstruct init pattern
- ✓ POJO fields match multifield child field names (e.g., itemText, itemNumber) NOT component field names (e.g., text)
- ✓ Creating getter methods for ALL fields (component + multifield)
- ✓ Multifield getter returns List<PojoClass>, NOT List<Resource>
- ✓ Including only getter methods (no setters unless pattern shows them)

**HTL GENERATION CHECKLIST:**
Before generating HTL, verify you are:
- ✓ Using data-sly-use to bind the Sling Model
- ✓ Displaying ALL component-level fields from "Fields to implement" (count them and verify!)
- ✓ Using ${{model.fieldName}} syntax for component fields
- ✓ For multifield: Using data-sly-list to iterate the POJO list
- ✓ Accessing multifield child fields via item POJO (e.g., ${{item.itemText}})
- ✓ Not confusing component field names with multifield child field names

**OUTPUT FORMAT:**
Return valid JSON only:
{{
  "dialog": "<complete dialog XML>",
  "sling_model": "<complete Java Sling Model>",
  "htl": "<complete HTL template>"
}}"""

    print("🤖 Generating code...")
    print(f"📊 Context lengths - Dialog: {len(rag_context['dialog_context'])} chars, Sling: {len(rag_context['sling_context'])} chars")
    
    try:
        # Use higher temperature for better adaptation while maintaining structure
        response = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.4,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert AEM developer. Generate code that implements user requirements exactly, using knowledge base examples as structural patterns. Prioritize user requirements over example patterns when they conflict. CRITICAL: Never omit fields from 'Fields to implement' list - all component-level fields must be included in Dialog XML, Sling Model (@ValueMapValue), and HTL (model.fieldName). If similar field types appear in both component fields and multifield context, include BOTH with different names: component field gets @ValueMapValue, multifield child goes in POJO only."
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
        
        # Validation
        if not dialog:
            return ("❌ Failed to generate dialog", "", "")
        if not sling_model:
            return ("❌ Failed to generate Sling Model", dialog, "")
        if not htl:
            return ("❌ Failed to generate HTL", dialog, sling_model)
        
        print("✅ Code generated successfully")
        return (dialog, sling_model, htl)
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {str(e)}")
        print(f"Raw output preview: {ai_output[:500]}")
        return (f"❌ JSON parsing error: {str(e)}\n\nRaw output:\n{ai_output[:500]}", "", "")
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return (f"❌ Error generating code: {str(e)}", "", "")
        
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

    # Event handlers
    generate_btn.click(
        fn=lambda user_ctx: generate_sling_model_with_rag(fields_data, user_ctx),
        inputs=[context_input],
        outputs=[dialog_output, sling_output, htl_output],
    )

    reset_btn.click(
        fn=reset_fields,
        inputs=[],
        outputs=[field_list, status, dialog_output, sling_output, htl_output],
    )

    add_btn.click(
        fn=add_field,
        inputs=[field_type, field_name, field_label, field_list],
        outputs=[field_list, status, gr.State(fields_data)],
    )


demo.launch(share=False)