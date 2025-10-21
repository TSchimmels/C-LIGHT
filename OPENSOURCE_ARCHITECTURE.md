# C-LIGHT Open-Source Architecture

## Vision
**C-LIGHT** (Cognitive, Life-science, Intelligence Gathering & Hypothesis Testing) is an open-source RAG system for analyzing quantum and molecular interactions in biological subjects. It harvests scientific papers, extracts causal relationships, builds knowledge graphs, and provides cross-scale interaction insights from quantum to organism level.

## Design Principles

1. **Standalone Excellence** - World-class biological interaction analysis RAG without external dependencies
2. **Plugin Architecture** - Can integrate with proprietary systems (e.g., CANDLE)
3. **Community-Driven** - Open contribution model for papers, weights, and algorithms
4. **Scientific Rigor** - Evidence-based weighting, citation tracking, reproducibility
5. **Multi-Scale Insights** - Discover interactions across quantum, molecular, cellular, tissue, and organism levels

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    C-LIGHT Core System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Harvesting │  │  Processing  │  │  Knowledge   │      │
│  │    Pipeline  │→ │   Pipeline   │→ │    Graph     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         ↓                  ↓                  ↓              │
│  ┌──────────────────────────────────────────────────┐      │
│  │         Causal Inference & Node Weighting         │      │
│  └──────────────────────────────────────────────────┘      │
│         │                                                    │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │      RAG System (Query & Insight Generation)      │      │
│  └──────────────────────────────────────────────────┘      │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Plugin Interface                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CANDLE     │  │   LangChain  │  │    Custom    │      │
│  │   Adapter    │  │   Adapter    │  │   Adapters   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Harvesting Pipeline
**Purpose**: Collect scientific papers from multiple sources

**Sources**:
- arXiv (cs.AI, cs.HC, q-bio.NC, q-bio.TO, q-bio.CB, q-bio.MN, quant-ph, physics.bio-ph, etc.)
- PubMed/PubMed Central (biomedical literature, organoid research, nutritional neuroscience)
- bioRxiv/medRxiv (preprints, organoid development, metabolic studies)
- PhilSci Archive (consciousness studies, philosophy of mind)
- OSF Preprints (Open Science Framework - consciousness research, organoid protocols)
- Cell Press (Cell Stem Cell, Stem Cell Reports - organoid research)
- Nature (Nature Protocols, Nature Methods - organoid methodologies)
- Community contributions (manual uploads, organoid protocols)

**Features**:
- DOI-based deduplication (RocksDB + SQLite)
- Multi-source aggregation
- Metadata extraction
- PDF processing
- Citation tracking

### 2. Processing Pipeline
**Purpose**: Extract knowledge and build relationships

**Extractors**:
- **Causal Relation Extractor**
  - Pattern-based extraction (causes, leads to, affects, etc.)
  - Dependency parsing
  - Confidence scoring
  - Context preservation

- **Behavioral Indicator Extractor**
  - Risk factors
  - Protective factors
  - Behavioral markers
  - Interventions

- **Domain Classifier**
  - Neuroscience
  - Psychology (Personality, Developmental, Clinical, Cognitive)
  - Developmental Psychology (Maturation, Life Stages, Psychosocial Development)
  - Personality Psychology (Big Five, MBTI, Enneagram, Trait Theory)
  - Humanistic Psychology (Maslow's Hierarchy, Self-Actualization, Human Potential)
  - Molecular Biology (Protein Interactions, Signaling Cascades, Molecular Mechanisms)
  - Pharmacology & Drug Interactions (Pharmacokinetics, Pharmacodynamics, Polypharmacy)
  - Epigenetics (DNA Methylation, Histone Modification, Gene-Environment Interactions)
  - Behavioral Neuroscience (Dopamine Systems, Reward Pathways, Motivation Circuits)
  - Sleep Science
  - Nutrition & Metabolic Cognition
  - Exercise Physiology
  - Microbiome
  - Social Psychology (Group Dynamics, Social Influence, Conformity, Tribalism)
  - Political Psychology (Polarization, Belief Formation, Propaganda, Ideology)
  - Cultural Neuroscience (Cultural Norms, Honor Cultures, Collective Behavior)
  - Digital Psychology (Social Media, Hypernudging, Persuasive Technology, Dark Patterns)
  - Sociology & Social Engineering
  - Behavioral Economics (Decision-Making, Nudging, Choice Architecture)
  - Quantum Physics
  - Consciousness Studies
  - Quantum Biology
  - EMF Biology (Electromagnetic Field Effects)
  - Biophysics
  - Quantum Cognition
  - Brain Organoid Biology
  - Tissue Engineering & Organoid Technology

- **Entity Extractor**
  - Biological entities (genes, proteins, neurotransmitters, microtubules, ion channels, stem cells, organoids, receptors, enzymes)
  - Molecular interactions (protein-protein binding, receptor-ligand, enzyme-substrate, signaling molecules)
  - Drugs and compounds (pharmaceuticals, natural compounds, agonists, antagonists, modulators)
  - Epigenetic markers (DNA methylation sites, histone modifications, chromatin states, transcription factors)
  - Psychological constructs (attention, memory, emotion, consciousness, awareness, self-actualization, beliefs, attitudes)
  - Personality traits (Big Five dimensions, MBTI types, temperament, character traits)
  - Developmental stages (infancy, childhood, adolescence, adulthood, maturation levels, Erikson's stages)
  - Motivational constructs (Maslow's hierarchy, needs, drives, self-determination, incentive salience, reward value)
  - Dopamine pathways (VTA, nucleus accumbens, mesolimbic, mesocortical, dopamine receptors D1-D5)
  - Social constructs (social norms, cultural values, in-groups, out-groups, social status, reputation, honor)
  - Digital manipulation techniques (hypernudging, dark patterns, infinite scroll, variable rewards, notifications, engagement metrics)
  - Cognitive biases (confirmation bias, motivated reasoning, availability heuristic, anchoring, framing effects)
  - Political constructs (ideology, partisanship, polarization, beliefs, moral foundations, sacred values)
  - Cultural factors (collectivism, individualism, honor culture, dignity culture, traditions, rituals)
  - Metabolic states (fasting, fed state, ketosis, glycolysis, metabolic switching)
  - Nutritional factors (glucose, ketones, vitamins, minerals, macronutrients, dietary patterns, methyl donors)
  - Organoid types (brain organoids, cerebral organoids, neural organoids, tissue models)
  - Quantum phenomena (coherence, entanglement, tunneling, superposition)
  - EMF parameters (frequency, intensity, exposure duration, field type)
  - Physical processes (quantum decoherence, wave function collapse, field interactions)
  - Interventions (drugs, therapies, lifestyle changes, EMF exposure protocols, dietary interventions, fasting protocols, social interventions)
  - Outcomes (performance, health, behavior, quantum states, field effects, cognitive function, personality development, belief change, social behavior)

### 3. Knowledge Graph
**Purpose**: Build interconnected knowledge base with weighted relationships

**Node Types**:
- **Concepts**: Abstract ideas (e.g., "cognitive performance", "neuroplasticity")
- **Entities**: Concrete things (e.g., "dopamine", "Mediterranean diet")
- **Papers**: Scientific documents
- **Interventions**: Actions/treatments
- **Outcomes**: Measured results

**Edge Types**:
- **Causal**: X causes Y (weighted by evidence strength)
- **Correlational**: X correlates with Y
- **Modulates**: X affects strength of Y
- **Predicts**: X predicts Y
- **Contradicts**: Paper A contradicts Paper B

**Weighting System**:
```python
Edge Weight = f(
    evidence_strength,    # Strong (meta-analysis) → Weak (single study)
    citation_count,       # Highly cited papers get more weight
    journal_impact,       # Nature > arXiv preprint
    replication_status,   # Replicated findings weighted higher
    sample_size,          # Larger N = higher confidence
    study_design,         # RCT > observational
    publication_date,     # Recent studies weighted slightly higher
    community_votes       # Upvotes/downvotes from users
)
```

**Community Features**:
- Users can propose weight adjustments
- Evidence-based voting system
- Transparent weight calculation
- Version control for graph state

### 4. Causal Inference Engine
**Purpose**: Discover cross-domain insights and causal chains

**Algorithms**:
- **Path Finding**: Find causal chains (e.g., Sleep → Cortisol → Memory)
- **Mediation Analysis**: Identify mediating factors
- **Confounding Detection**: Identify potential confounders
- **Effect Size Estimation**: Aggregate effect sizes across studies
- **Contradiction Resolution**: Handle conflicting evidence

**Query Examples**:
```python
# Find what affects cognitive performance
kg.query_influences("cognitive performance", max_depth=3)

# Find causal path
kg.find_causal_path("sleep deprivation", "anxiety", max_hops=4)

# Cross-domain insights
kg.cross_domain_effects(
    source_domains=["nutrition", "microbiome"],
    target_domains=["psychology", "neuroscience"]
)

# Cross-domain quantum biology insights
kg.cross_domain_effects(
    source_domains=["quantum_biology", "emf_biology"],
    target_domains=["neuroscience", "consciousness_studies"]
)

# EMF effects on cognition
kg.find_causal_path("electromagnetic_fields", "cognitive_performance", max_hops=5)

# Quantum effects in consciousness
kg.query_influences("quantum_coherence", domains=["consciousness_studies", "quantum_biology"])

# Intervention analysis
kg.intervention_outcomes("intermittent fasting")

# Personality and cognitive function
kg.query_influences("personality_traits", domains=["psychology", "neuroscience"])

# Developmental maturation effects
kg.find_causal_path("developmental_stage", "cognitive_capacity", max_hops=4)

# Metabolic states and cognition
kg.cross_domain_effects(
    source_domains=["nutrition", "metabolism"],
    target_domains=["cognition", "psychology", "neuroscience"]
)

# Fasting effects on brain function
kg.find_causal_path("fasting", "cognitive_performance", max_hops=5)

# Glucose/sugar effects on cognition
kg.query_influences("glucose_metabolism", domains=["neuroscience", "metabolism", "psychology"])

# Maslow's hierarchy and self-actualization
kg.query_influences("self_actualization", domains=["humanistic_psychology", "developmental_psychology"])

# Organoid-based insights
kg.cross_domain_effects(
    source_domains=["organoid_biology", "tissue_engineering"],
    target_domains=["neuroscience", "developmental_biology", "disease_modeling"]
)

# Drug interactions and molecular effects
kg.query_influences("drug_interactions", domains=["pharmacology", "molecular_biology"])

# Epigenetic influences on behavior
kg.find_causal_path("epigenetic_modification", "behavioral_change", max_hops=6)

# Social media manipulation pathways
kg.find_causal_path("social_media_algorithm", "dopamine_release", max_hops=4)
kg.find_causal_path("hypernudging", "belief_change", max_hops=5)

# Cultural influences on gene expression
kg.cross_domain_effects(
    source_domains=["cultural_neuroscience", "social_psychology"],
    target_domains=["epigenetics", "gene_expression", "neuroscience"]
)

# Political polarization mechanisms
kg.find_causal_path("echo_chamber", "political_polarization", max_hops=5)

# Upstream molecular to downstream social
kg.find_causal_path("protein_interaction", "social_behavior", max_hops=10)
```

### 5. RAG System
**Purpose**: Answer questions with evidence-based insights

**Features**:
- **Semantic Search**: Vector similarity search across papers
- **Graph-Augmented Retrieval**: Use knowledge graph to find relevant papers
- **Multi-hop Reasoning**: Follow causal chains
- **Evidence Synthesis**: Aggregate findings across papers
- **Citation Generation**: Provide paper citations for all claims
- **Confidence Scores**: Indicate certainty of answers

**Example Query**:
```
User: "How does omega-3 affect depression?"

C-LIGHT Response:
Based on 47 papers in the knowledge graph:

1. Direct Effects (Strong Evidence):
   - Omega-3 fatty acids reduce inflammatory markers (IL-6, CRP) [12 studies]
   - EPA/DHA supplementation shows moderate effect on depressive symptoms (d=0.38) [Meta-analysis, 2023]

2. Causal Pathways (Moderate Evidence):
   - Omega-3 → ↓ Inflammation → ↓ Depression (5 mechanistic studies)
   - Omega-3 → ↑ Neuroplasticity → ↑ Mood (3 studies)
   - Omega-3 → ↑ Gut microbiome diversity → ↓ Depression (emerging, 2 studies)

3. Moderating Factors:
   - Effect stronger in clinical depression vs. subclinical
   - EPA more effective than DHA for mood
   - Dosage: 1-2g/day optimal

4. Contradictory Evidence:
   - 3 studies found no effect (possible reasons: low dosage, short duration)

Citations: [Shows relevant papers]
Confidence: 78% (based on study quality, replication, sample sizes)
```

**Example Query (Quantum Biology & Consciousness)**:
```
User: "What evidence exists for quantum effects in consciousness and cognition?"

C-LIGHT Response:
Based on 23 papers across quantum biology, consciousness studies, and neuroscience:

1. Quantum Biology Mechanisms (Emerging Evidence):
   - Quantum coherence in microtubules (Orch-OR theory) [4 theoretical papers, 2 experimental]
   - Quantum tunneling in enzyme reactions affecting neurotransmitter synthesis [3 studies]
   - Entanglement in photosynthetic complexes as biological quantum model [5 papers]

2. EMF Effects on Neural Function (Moderate Evidence):
   - Extremely low frequency (ELF) EMF affects calcium ion channels → neural excitability [8 studies]
   - Radiofrequency EMF modulates cognitive performance (mixed results) [12 studies]
   - Geomagnetic fields influence circadian rhythms via cryptochrome proteins [4 studies]

3. Cross-Domain Pathways (Speculative):
   - Quantum coherence → Microtubule function → Neural information processing → Consciousness
   - EMF exposure → Quantum state disruption → Cognitive effects (theoretical framework)

4. Critical Gaps:
   - Limited experimental evidence for quantum coherence at biological temperatures
   - Decoherence timescales vs. neural processing timescales (ongoing debate)
   - Lack of direct causal links between quantum effects and subjective experience

5. Contradictory Evidence:
   - Classical models adequately explain most neural phenomena without quantum mechanics
   - Room temperature quantum effects remain controversial in neuroscience

Citations: [Shows relevant papers from quant-ph, q-bio.NC, consciousness studies]
Confidence: 35% (emerging field, limited replication, theoretical frameworks dominant)
Research Status: Active area of investigation, requires more experimental validation
```

**Example Query (Metabolic States & Cognition)**:
```
User: "How does fasting affect cognitive performance and brain function?"

C-LIGHT Response:
Based on 64 papers across nutrition, neuroscience, metabolism, and psychology:

1. Metabolic Switching Effects (Strong Evidence):
   - Fasting induces ketosis → brain uses ketones as fuel [18 studies]
   - Intermittent fasting increases BDNF (brain-derived neurotrophic factor) [12 studies]
   - Ketones provide more efficient energy than glucose for neurons (d=0.42) [Meta-analysis, 2022]

2. Cognitive Performance (Moderate Evidence):
   - Short-term fasting (16-24h) → Improved focus and mental clarity [8 studies]
   - Long-term intermittent fasting → Enhanced memory consolidation [6 studies]
   - Glucose restriction → Reduced inflammation → Better cognitive function [9 studies]

3. Causal Pathways:
   - Fasting → ↑ Ketones → ↑ Mitochondrial function → ↑ Neuronal energy → ↑ Cognition
   - Fasting → ↓ Insulin → ↓ Inflammation → ↑ Synaptic plasticity
   - Fasting → ↑ Autophagy → Neural cleanup → Enhanced brain health

4. Moderating Factors:
   - Individual metabolic flexibility varies (genetic factors)
   - Adaptation period needed (2-4 weeks for metabolic switching)
   - Timing matters: cognitive benefits stronger after adaptation
   - Blood sugar regulation affects response

5. Contradictory Evidence:
   - Some studies show impaired performance during initial fasting (adaptation phase)
   - Hypoglycemia can impair cognition in non-adapted individuals [5 studies]

Citations: [Shows papers from q-bio.MN, q-bio.NC, nutrition journals]
Confidence: 72% (good replication, mechanistic understanding, some individual variation)
```

**Example Query (Brain Organoids & Development)**:
```
User: "What can brain organoids tell us about human neural development and disease?"

C-LIGHT Response:
Based on 89 papers across tissue engineering, neuroscience, and developmental biology:

1. Developmental Modeling (Strong Evidence):
   - Brain organoids recapitulate early cortical development [23 studies]
   - Cell type diversity matches fetal brain tissue [15 studies, scRNA-seq validation]
   - Synaptic maturation occurs over 6-9 months in culture [12 studies]
   - Spontaneous neural activity emerges by 2-3 months [18 studies, MEA recordings]

2. Disease Modeling Applications (Strong Evidence):
   - Autism spectrum disorders: synaptic overgrowth in patient-derived organoids [11 studies]
   - Alzheimer's disease: Aβ accumulation and tau pathology in organoids [9 studies]
   - Microcephaly: reduced organoid size matches patient phenotypes [14 studies]
   - Schizophrenia: altered neural oscillations in patient organoids [6 studies]

3. Metabolic Effects on Organoid Development (Emerging Evidence):
   - Glucose concentration affects organoid size and maturation [7 studies]
   - Ketone supplementation enhances mitochondrial function [4 studies]
   - Nutrient deprivation impacts neural differentiation [5 studies]

4. Cross-Domain Insights:
   - Organoids → Understanding human-specific neurodevelopment
   - Drug screening → Personalized medicine approaches
   - Electrophysiology → Neural circuit formation mechanisms
   - 3D culture → More physiologically relevant than 2D

5. Current Limitations:
   - Lack of vascularization limits organoid size and maturation
   - No immune system representation
   - Limited glia-neuron interactions compared to in vivo
   - Batch-to-batch variability remains a challenge

Citations: [Shows papers from q-bio.TO, q-bio.CB, Cell Stem Cell, Nature]
Confidence: 85% (rapidly advancing field, strong replication, clear applications)
Future Directions: Vascularized organoids, assembloids, multi-organoid systems
```

**Example Query (Personality & Developmental Psychology)**:
```
User: "How do personality traits develop across the lifespan and affect cognitive aging?"

C-LIGHT Response:
Based on 52 papers across developmental psychology, personality psychology, and neuroscience:

1. Personality Development Patterns (Strong Evidence):
   - Big Five traits show characteristic changes across lifespan [15 longitudinal studies]
   - Conscientiousness increases with age (peak in 50s-60s) [12 studies]
   - Openness to experience peaks in early adulthood, declines after 60 [9 studies]
   - Emotional stability increases from adolescence through middle age [11 studies]

2. Maturation and Self-Actualization (Moderate Evidence):
   - Higher needs in Maslow's hierarchy become salient with maturity [7 studies]
   - Self-actualization correlates with age and life experience [8 studies]
   - Developmental stage affects cognitive priorities and processing styles [6 studies]

3. Cognitive Aging Effects:
   - High conscientiousness → Slower cognitive decline [9 studies, d=0.31]
   - Openness → Maintained cognitive flexibility in aging [7 studies]
   - Neuroticism → Accelerated cognitive decline (stress pathway) [8 studies]

4. Causal Pathways:
   - Personality → Lifestyle choices → Brain health → Cognitive aging
   - Conscientiousness → Health behaviors → Reduced inflammation → Better cognition
   - Openness → Cognitive engagement → Neural plasticity → Cognitive reserve

5. Individual Differences:
   - Personality plasticity decreases with age but remains present
   - Life events can shift personality trajectories
   - Genetic factors account for 40-50% of personality stability

Citations: [Shows papers from developmental psychology, personality journals]
Confidence: 76% (good longitudinal data, replication, complex interactions)
```

**Example Query (Neuronal Firing Thresholds & Metabolic Modulation)**:
```
User: "How do nutritional states like fasting and glucose levels affect neuronal firing thresholds, and what can brain organoids tell us about this?"

C-LIGHT Response:
Based on 71 papers across neuroscience, metabolism, biophysics, and organoid biology:

1. Metabolic Effects on Neuronal Excitability (Strong Evidence):
   - Glucose availability affects ATP-dependent Na+/K+ pumps → alters resting potential [14 studies]
   - Ketones modulate KATP channels → hyperpolarization → increased firing threshold [9 studies]
   - Fasting increases neuronal excitability via BDNF-mediated ion channel modulation [11 studies]
   - Low glucose → decreased action potential amplitude and firing rate [8 studies]

2. Ion Channel Mechanisms (Strong Evidence):
   - Metabolic state regulates voltage-gated sodium channels (Nav1.1, Nav1.6) [12 studies]
   - Glucose deprivation opens KATP channels → membrane hyperpolarization [10 studies]
   - Ketone bodies (β-hydroxybutyrate) modulate calcium channels [7 studies]
   - Fasting increases expression of Kv channels → altered excitability [6 studies]

3. Brain Organoid Evidence (Emerging Evidence):
   - Glucose concentration (5-25mM) affects organoid firing patterns [8 studies, MEA data]
   - Ketone supplementation increases spontaneous activity in mature organoids [4 studies]
   - Nutrient deprivation alters network synchronization in organoids [5 studies]
   - Metabolic switching observable in organoid electrophysiology [3 studies]

4. Causal Pathways:
   - Fasting → ↑ Ketones → KATP channel modulation → Altered threshold → Network reconfiguration
   - Glucose ↓ → ATP ↓ → Na+/K+ pump efficiency ↓ → Depolarized resting potential → Reduced excitability
   - Metabolic state → Ion channel expression → Firing threshold → Cognitive performance
   - Nutrition → Energy availability → Neuronal firing → Brain oscillations → Cognitive state

5. Practical Implications:
   - Intermittent fasting may optimize neuronal excitability through metabolic switching
   - Blood sugar stability important for consistent neuronal firing
   - Individual metabolic flexibility affects response to dietary interventions
   - Organoids provide platform to test nutritional interventions on human neurons

6. Cross-Domain Insights:
   - Organoid models validate human-specific metabolic effects on firing
   - Metabolic modulation could be therapeutic for epilepsy (threshold manipulation)
   - Cognitive performance correlates with optimal metabolic-excitability balance
   - EMF effects may interact with metabolic state to affect thresholds

7. Current Limitations & Gaps:
   - Limited long-term organoid studies under varied metabolic conditions
   - Individual genetic variation in ion channel responses not fully characterized
   - Complex interactions between multiple metabolic factors need mapping
   - Translation from organoid to in vivo human brain needs validation

Citations: [Shows papers from q-bio.NC, q-bio.MN, q-bio.TO, physics.bio-ph, biophysics journals]
Confidence: 71% (good mechanistic understanding, organoid data emerging, translation pending)
Research Frontiers: Long-term metabolic organoid studies, personalized threshold profiling, therapeutic applications
```

**Example Query (Social Media Manipulation & Dopamine Hijacking)**:
```
User: "How do social media algorithms manipulate dopamine systems to influence behavior and beliefs?"

C-LIGHT Response:
Based on 94 papers across neuroscience, digital psychology, behavioral economics, and social sciences:

1. Dopamine Manipulation Mechanisms (Strong Evidence):
   - Variable reward schedules trigger dopamine release → craving for next reward [18 studies]
   - Unpredictable "likes" and notifications activate nucleus accumbens more than predictable rewards [12 studies]
   - Social validation (likes, shares) activates same dopamine pathways as drugs [15 studies, fMRI data]
   - Infinite scroll exploits dopamine-driven exploratory behavior [9 studies]

2. Hypernudging & Dark Patterns (Strong Evidence):
   - Default settings exploit status quo bias to maximize engagement [14 studies]
   - Scarcity cues ("only 2 people watching") trigger urgency via dopamine [8 studies]
   - Social proof ("trending") hijacks conformity instincts [11 studies]
   - Variable notification timing optimizes dopamine response [7 studies]

3. Algorithmic Manipulation Pathways (Emerging Evidence):
   - Recommendation algorithms → content selection → dopamine hit → belief reinforcement [13 studies]
   - Echo chambers increase prediction accuracy → dopamine from correct predictions → polarization [9 studies]
   - Outrage content triggers stronger dopamine response than neutral content [8 studies]
   - Personalized feeds exploit individual dopamine sensitivities [6 studies]

4. Causal Chain to Belief Formation:
   - Algorithm → Dopamine manipulation → Increased engagement → Filter bubble → Echo chamber → Belief polarization
   - Variable rewards → Compulsive checking → Reduced critical thinking → Belief adoption
   - Social validation → Dopamine hit → Associative learning → Value assignment to beliefs
   - Outrage content → Dopamine + cortisol → Emotional arousal → Memory consolidation → Belief strengthening

5. Individual Vulnerabilities:
   - Low baseline dopamine (ADHD, depression) more susceptible to manipulation [10 studies]
   - Adolescent brains (immature prefrontal cortex) especially vulnerable [12 studies]
   - High novelty-seeking personality trait increases susceptibility [7 studies]
   - Social isolation increases vulnerability to social validation manipulation [8 studies]

6. Downstream Consequences:
   - Digital addiction (compulsive use, withdrawal) [15 studies]
   - Attention fragmentation and cognitive decline [11 studies]
   - Political polarization via echo chambers [14 studies]
   - Anxiety and depression from social comparison [13 studies]
   - Belief radicalization through algorithmic funneling [9 studies]

7. Resistance Mechanisms:
   - Awareness of manipulation reduces susceptibility [6 studies]
   - Notification management reduces dopamine hijacking [8 studies]
   - Deliberate diverse content exposure counteracts echo chambers [7 studies]
   - Metacognitive training improves resistance [5 studies]

Citations: [Shows papers from cs.CY, cs.HC, q-bio.NC, behavioral economics, digital psychology]
Confidence: 79% (strong evidence for dopamine mechanisms, emerging evidence for long-term belief effects)
Societal Impact: Critical for understanding modern information warfare and cognitive autonomy
```

**Example Query (Epigenetics → Behavior Causal Chain)**:
```
User: "How do environmental factors trigger epigenetic changes that influence temperament, emotion, and personality?"

C-LIGHT Response:
Based on 78 papers across epigenetics, developmental neuroscience, and behavioral psychology:

1. Environmental Triggers (Strong Evidence):
   - Early life stress → HPA axis activation → cortisol → epigenetic modifications [22 studies]
   - Childhood trauma → DNA methylation changes in stress-response genes (NR3C1, FKBP5) [16 studies]
   - Maternal care quality → methylation of oxytocin and dopamine receptor genes [14 studies]
   - Nutritional factors (methyl donors) → global DNA methylation changes [12 studies]
   - Social isolation → histone modifications in social behavior genes [9 studies]

2. Specific Epigenetic Mechanisms (Strong Evidence):
   - DNA methylation at gene promoters → reduced gene expression → altered protein levels [all studies]
   - Histone acetylation → chromatin opening → increased transcription [18 studies]
   - MicroRNA regulation → post-transcriptional gene silencing [11 studies]
   - Transgenerational epigenetic inheritance (3+ generations observed) [8 studies in animals, emerging in humans]

3. Gene Expression → Neurobiology (Strong Evidence):
   - Serotonin transporter (5-HTTLPR) methylation → reduced serotonin signaling → anxiety temperament [13 studies]
   - BDNF methylation → reduced neuroplasticity → depression vulnerability [11 studies]
   - Dopamine receptor methylation → altered reward sensitivity → personality traits [9 studies]
   - Oxytocin receptor methylation → reduced social bonding → attachment style [10 studies]

4. Neurobiology → Behavioral Outcomes (Strong Evidence):
   - Epigenetically-modified stress genes → heightened amygdala reactivity → anxiety [14 studies]
   - Altered dopamine genes → reward system changes → novelty-seeking behavior [10 studies]
   - Modified serotonin genes → mood regulation impairment → emotional temperament [12 studies]
   - Changed oxytocin genes → social behavior alterations → personality traits [8 studies]

5. Complete Causal Chain Examples:
   - Childhood neglect → NR3C1 methylation → reduced cortisol receptor → HPA dysregulation → anxiety temperament
   - High maternal care → reduced BDNF methylation → increased neuroplasticity → resilient personality
   - Fasting/diet → altered methyl donor availability → global methylation changes → metabolic gene expression → energy and mood
   - Social stress → oxytocin receptor methylation → reduced social bonding → avoidant personality

6. Critical Windows:
   - Prenatal period: highest epigenetic plasticity [all studies]
   - Early childhood (0-5 years): major epigenetic programming [18 studies]
   - Adolescence: second wave of epigenetic remodeling [12 studies]
   - Adult plasticity: present but reduced [9 studies]

7. Reversibility and Interventions:
   - Environmental enrichment can reverse some epigenetic marks [11 studies in animals, 4 in humans]
   - Therapeutic interventions (psychotherapy) show epigenetic changes [6 studies]
   - Nutritional interventions (methyl donors) modulate methylation [8 studies]
   - Exercise affects epigenetic markers [9 studies]

8. Individual Variation:
   - Genetic background affects epigenetic susceptibility [13 studies]
   - Sex differences in epigenetic responses [10 studies]
   - Developmental timing determines permanence [all studies]

Citations: [Shows papers from q-bio.GN, q-bio.NC, developmental psychology, molecular psychiatry]
Confidence: 82% (strong mechanistic understanding, human studies emerging, causality established in animals)
Clinical Implications: Epigenetic markers as biomarkers, early intervention windows, personalized approaches
```

**Example Query (Drug Interactions & Cognitive Effects)**:
```
User: "How do drug-drug and drug-nutrient interactions affect neurotransmitter systems and cognitive performance?"

C-LIGHT Response:
Based on 103 papers across pharmacology, neuroscience, and clinical medicine:

1. Pharmacokinetic Interactions (Strong Evidence):
   - CYP450 enzyme competition → altered drug metabolism → unexpected concentrations [45 studies]
   - P-glycoprotein interactions → altered blood-brain barrier permeability [18 studies]
   - Protein binding displacement → increased free drug concentration [12 studies]
   - Food-drug interactions alter absorption (30-70% variation) [22 studies]

2. Pharmacodynamic Interactions (Strong Evidence):
   - Synergistic effects at receptor level (e.g., benzodiazepines + alcohol) [28 studies]
   - Antagonistic effects reducing efficacy [15 studies]
   - Neurotransmitter depletion from multiple drugs [11 studies]
   - Downstream signaling interference [14 studies]

3. Common Neurotransmitter System Interactions:
   - SSRIs + NSAIDs → serotonin syndrome risk [12 studies]
   - Stimulants + antidepressants → dopamine/norepinephrine synergy [9 studies]
   - Statins + CoQ10 depletion → mitochondrial dysfunction → cognitive effects [13 studies]
   - Proton pump inhibitors + B12/magnesium depletion → neurological effects [16 studies]

4. Drug-Nutrient Interactions Affecting Cognition:
   - Caffeine + L-theanine → enhanced attention without jitter [8 studies]
   - Alcohol + thiamine depletion → Wernicke-Korsakoff syndrome [all studies]
   - Metformin + B12 depletion → cognitive impairment in elderly [10 studies]
   - Statins + fat-soluble vitamin depletion → potential cognitive effects [9 studies]

5. Polypharmacy Effects (Strong Evidence):
   - >5 medications → exponential interaction risk [18 studies]
   - Anticholinergic burden → cognitive decline [14 studies]
   - Combined sedative effects → attention and memory impairment [12 studies]
   - Drug-induced nutrient depletion accumulation [11 studies]

6. Molecular Mechanisms → Cognitive Outcomes:
   - Receptor competition → altered neurotransmission → cognitive function changes
   - Enzyme inhibition → accumulation → side effects → cognitive impairment
   - Nutrient depletion → cofactor shortage → impaired neurotransmitter synthesis
   - Mitochondrial effects → energy depletion → neuronal dysfunction

7. Vulnerable Populations:
   - Elderly (altered pharmacokinetics, polypharmacy) [20 studies]
   - Genetic polymorphisms (CYP450 variants) → variable responses [15 studies]
   - Liver/kidney disease → impaired clearance [all studies]
   - Poor nutritional status → increased vulnerability [9 studies]

8. Clinical Implications:
   - Comprehensive medication reviews essential [clinical guidelines]
   - Nutrient status monitoring in chronic medication users [emerging practice]
   - Personalized medicine based on genetic testing [growing evidence, 12 studies]
   - Timing of medications and meals affects outcomes [14 studies]

Citations: [Shows papers from pharmacology, clinical medicine, neuroscience, nutrition journals]
Confidence: 88% (well-established in pharmacology, growing evidence for cognitive effects)
Clinical Urgency: High - affects millions on multiple medications
```

**Example Query (Temporal Dynamics: Caffeine, Sugar, Food Timing)**:
```
User: "What is the complete time course for caffeine and sugar, including onset, peak, crash, and how does food timing affect these dynamics?"

C-LIGHT Response:
Based on 127 papers across pharmacology, metabolism, neuroscience, and chronobiology:

1. Caffeine Temporal Profile (Strong Evidence):
   **ONSET** (15-45 minutes):
   - Absorption begins in stomach, peaks in small intestine [18 studies]
   - Blood-brain barrier penetration within 15-20 minutes [12 studies]
   - Adenosine receptor blockade initiates [all studies]
   - Food delays absorption by 20-40 minutes [14 studies]

   **PEAK** (1-2 hours):
   - Peak plasma concentration at 60-120 minutes [28 studies]
   - Maximum adenosine receptor occupancy [15 studies]
   - Peak dopamine and norepinephrine release [14 studies]
   - Peak cortisol elevation (30-50% increase) [11 studies]
   - Alertness, focus, and performance peak [all studies]

   **HALF-LIFE** (3-7 hours, highly variable):
   - Average: 5 hours in healthy adults [32 studies]
   - Genetic variation (CYP1A2): 2hr (fast metabolizers) to 9hr (slow) [16 studies]
   - Oral contraceptives double half-life [8 studies]
   - Pregnancy triples half-life (10-18 hours) [7 studies]
   - Smoking reduces half-life by 50% [9 studies]

   **CRASH** (6-12 hours post-ingestion):
   - Adenosine rebound (accumulated adenosine floods receptors) [18 studies]
   - Dopamine/norepinephrine drop below baseline [12 studies]
   - Fatigue, irritability, difficulty concentrating [all studies]
   - Headache in regular users (withdrawal symptom) [14 studies]
   - Sleep disruption if consumed after 2pm [22 studies]

   **TOLERANCE** (1-7 days):
   - Adenosine receptor upregulation within 1 week [11 studies]
   - Reduced subjective alertness effects [15 studies]
   - Physical dependence after 3-7 consecutive days [9 studies]
   - Withdrawal headache peaks 24-48hr after cessation [all studies]

2. Sugar/Glucose Temporal Profile (Strong Evidence):
   **ONSET** (5-15 minutes):
   - Glucose absorption begins in mouth (sublingual) [8 studies]
   - Rapid stomach emptying of simple sugars [12 studies]
   - Blood glucose starts rising within 10 minutes [all studies]
   - Insulin secretion triggered (first phase) [18 studies]

   **SPIKE** (15-45 minutes):
   - Peak blood glucose at 30-60 minutes (food dependent) [35 studies]
   - Peak insulin at 30-90 minutes [28 studies]
   - Dopamine release from glucose sensing neurons [14 studies]
   - Temporary energy and mood boost [all studies]
   - Glycemic index affects spike magnitude (high GI = faster, higher) [42 studies]

   **CLEARANCE** (1-2 hours):
   - Insulin drives glucose into cells [all studies]
   - Blood glucose rapidly declines [28 studies]
   - Overcorrection common with high insulin response [16 studies]

   **CRASH** (2-4 hours - Reactive Hypoglycemia):
   - Blood glucose drops below baseline (70-60 mg/dL) [24 studies]
   - Counter-regulatory hormones release (cortisol, adrenaline) [18 studies]
   - Symptoms: fatigue, irritability, anxiety, shakiness, brain fog [all studies]
   - Strong cravings for more sugar (addiction-like cycle) [14 studies]
   - Impaired decision-making and self-control [12 studies]

   **CHRONIC EFFECTS** (weeks-months):
   - Insulin resistance development [32 studies]
   - Dopamine receptor downregulation → reduced reward sensitivity [15 studies]
   - Altered hunger hormones (leptin, ghrelin) [18 studies]
   - Increased sugar cravings and tolerance [16 studies]

3. Food Timing Effects on Caffeine (Strong Evidence):
   **FASTED STATE**:
   - Faster absorption (peak at 45-60 min) [12 studies]
   - Higher peak concentration (30-40% higher) [14 studies]
   - More intense effects (positive and negative) [9 studies]
   - Increased jitteriness and anxiety in sensitive individuals [11 studies]
   - Enhanced fat oxidation when combined with exercise [8 studies]

   **FED STATE**:
   - Slower absorption (peak at 90-120 min) [12 studies]
   - Lower peak concentration [14 studies]
   - Prolonged duration of effects [8 studies]
   - Reduced GI side effects [7 studies]
   - Food composition matters: Fat > Protein > Carbs for delay [9 studies]

4. Caffeine + Sugar Combination Effects (Strong Evidence):
   **SYNERGISTIC EFFECTS**:
   - Combined dopamine spike (additive effect) [13 studies]
   - Enhanced subjective energy (greater than either alone) [11 studies]
   - Improved cognitive performance short-term [16 studies]
   - Energy drinks exploit this synergy [clinical observations]

   **SYNERGISTIC CRASH**:
   - Dual mechanism crash (adenosine rebound + hypoglycemia) [9 studies]
   - More severe fatigue and irritability [7 studies]
   - Stronger rebound cravings [6 studies]
   - Vicious cycle: crash → caffeine/sugar → temporary relief → worse crash [emerging evidence]

   **LONG-TERM EFFECTS**:
   - Faster insulin resistance development [8 studies]
   - Amplified dopamine receptor downregulation [6 studies]
   - Disrupted hunger/satiety signaling [7 studies]
   - Increased anxiety and sleep disruption [12 studies]

5. Optimal Timing Strategies (Moderate Evidence):
   **CAFFEINE**:
   - Morning timing: 9-11:30am optimal (after cortisol awakening response) [11 studies]
   - Avoid after 2pm for sleep quality [18 studies]
   - With food for stable effects [9 studies]
   - Cycling: 2 weeks on, 1 week off prevents full tolerance [6 studies]

   **SUGAR/CARBS**:
   - With protein/fat to blunt insulin spike [22 studies]
   - Post-workout window: beneficial for glycogen replenishment [18 studies]
   - Earlier in day: better insulin sensitivity morning vs. evening [14 studies]
   - Low-glycemic choices prevent reactive hypoglycemia [28 studies]

6. Individual Variation (Strong Evidence):
   **GENETIC FACTORS**:
   - CYP1A2 polymorphisms → 4-fold variation in caffeine metabolism [16 studies]
   - Insulin sensitivity varies widely [24 studies]
   - Adenosine receptor variants affect caffeine sensitivity [9 studies]

   **STATE FACTORS**:
   - Sleep debt amplifies caffeine effects and extends half-life [12 studies]
   - Stress affects both caffeine and glucose metabolism [14 studies]
   - Hormonal state (menstrual cycle, pregnancy) affects metabolism [11 studies]
   - Liver function affects clearance [all studies]

7. Practical Implications (Clinical Evidence):
   - Timing caffeine with food reduces jitteriness and extends duration
   - Avoiding caffeine after 2pm preserves sleep quality
   - Combining protein with carbs prevents glucose crash
   - Awareness of individual tolerance and metabolism patterns
   - Strategic timing improves cognitive performance and mood stability
   - Breaking caffeine/sugar dependency requires understanding withdrawal timelines

8. Withdrawal Timelines (Strong Evidence):
   **CAFFEINE WITHDRAWAL**:
   - Onset: 12-24 hours after last dose [all studies]
   - Peak: 24-48 hours (severe headache, fatigue, irritability) [18 studies]
   - Duration: 2-9 days for acute symptoms [15 studies]
   - Protracted: Subtle effects up to 21 days [6 studies]

   **SUGAR WITHDRAWAL**:
   - Cravings peak: 2-5 days [12 studies]
   - Mood and energy stabilize: 1-2 weeks [9 studies]
   - Insulin sensitivity improves: 2-4 weeks [16 studies]
   - Dopamine sensitivity normalizes: 4-12 weeks [7 studies]

Citations: [Shows papers from pharmacology, chronobiology, metabolism, neuroscience journals]
Confidence: 91% (extensive pharmacokinetic data, well-characterized temporal profiles, strong clinical evidence)
Practical Value: Critical for optimizing cognitive performance and avoiding crashes
Individual Variation: High - genetic and lifestyle factors create 2-4x variation in responses
```

## Plugin Interface

### Base Interface
```python
class CognitiveScienceRAGInterface(ABC):
    """Interface for integrating C-LIGHT with external systems"""

    @abstractmethod
    def add_document(self, title: str, content: str, metadata: Dict) -> str:
        """Add a document to the RAG system"""
        pass

    @abstractmethod
    def query(self, question: str, **kwargs) -> Dict:
        """Query the RAG system"""
        pass

    @abstractmethod
    def get_causal_chain(self, source: str, target: str) -> List[Dict]:
        """Get causal pathway between concepts"""
        pass

    @abstractmethod
    def update_node_weight(self, node_id: str, weight: float, reason: str):
        """Update knowledge graph node weight"""
        pass

    @abstractmethod
    def export_knowledge_graph(self, format: str = "json") -> Any:
        """Export knowledge graph for external use"""
        pass
```

### CANDLE Adapter
```python
class CANDLEAdapter(CognitiveScienceRAGInterface):
    """Adapter for integrating C-LIGHT with CANDLE explainatory system"""

    def __init__(self, candle_rag: 'CognitiveScienceRAG'):
        self.candle_rag = candle_rag
        self.clight_kg = CLightKnowledgeGraph()

    def sync_to_candle(self):
        """Push C-LIGHT knowledge graph to CANDLE"""
        for node in self.clight_kg.nodes:
            self.candle_rag.add_concept(node)

    def sync_from_candle(self):
        """Pull insights from CANDLE's explainatory system"""
        # CANDLE may have proprietary reasoning
        pass
```

## Data Storage

### Architecture
```
/data/
├── papers/
│   ├── raw/           # Downloaded PDFs (HDD)
│   ├── processed/     # Extracted text, metadata (NVMe)
│   └── embeddings/    # Vector embeddings (NVMe)
├── databases/
│   ├── doi_db/        # RocksDB + SQLite (NVMe)
│   ├── vector_store/  # FAISS or Qdrant (NVMe)
│   └── graph_db/      # Neo4j or NetworkX (NVMe)
└── models/
    ├── embeddings/    # SentenceTransformer models
    └── extractors/    # Fine-tuned extraction models
```

### Database Schema

**DOI Database** (SQLite):
- Papers table: DOI, title, authors, abstract, categories, dates
- Processing table: Extraction results, quality scores
- Citations table: Citation network

**Knowledge Graph** (Neo4j or NetworkX):
- Nodes: Concepts, entities, papers, interventions
- Edges: Causal, correlational, contradicts
- Properties: Weights, confidence, evidence

**Vector Store** (FAISS):
- Paper embeddings
- Concept embeddings
- Fast semantic search

## Community Contribution Model

### 1. Paper Contributions
- Users can submit papers (with DOI or PDF)
- Automated quality checks
- Community review process
- Credit tracking for contributors

### 2. Weight Adjustments
- Users propose weight changes with justification
- Evidence-based voting (reputation-weighted)
- Transparent change log
- Versioned graph snapshots

### 3. Extraction Improvements
- Community can improve extraction patterns
- Submit better causal patterns
- Train domain-specific extractors
- Share fine-tuned models

### 4. Domain Expertise
- Domain experts can curate subgraphs
- Verify causal relationships
- Add missing connections
- Flag contradictions

### 5. Quality Metrics
- Contribution reputation score
- Paper quality ratings
- Extraction accuracy feedback
- Graph consistency checks

## Deployment Options

### 1. Self-Hosted (Full Control)
```bash
# Download all papers and build local knowledge graph
docker-compose up c-light-full
```

### 2. RAG-Only (Lightweight)
```bash
# Just the RAG system, no harvesting
pip install c-light-rag
```

### 3. Cloud-Hosted (Community Instance)
```
# Connect to community-maintained instance
c-light connect --instance community.c-light.org
```

### 4. CANDLE Integration (Classified)
```python
from c_light import CLightKnowledgeGraph
from candle import CognitiveScienceRAG

# Use C-LIGHT as knowledge source for CANDLE
clight = CLightKnowledgeGraph()
candle_adapter = CANDLEAdapter(clight, candle_rag)
candle_adapter.sync_to_candle()
```

## Roadmap

### Phase 1: Core Infrastructure (Months 1-3)
- [ ] Abstract CANDLE dependencies
- [ ] Implement standalone knowledge graph
- [ ] Build node weighting system
- [ ] Create basic RAG interface
- [ ] Set up DOI database
- [ ] Implement arXiv harvesting

### Phase 2: Advanced Features (Months 4-6)
- [ ] Add PubMed/bioRxiv harvesting
- [ ] Implement causal inference engine
- [ ] Build community contribution system
- [ ] Create web interface
- [ ] Add visualization tools
- [ ] Develop CANDLE adapter

### Phase 3: Community & Scale (Months 7-12)
- [ ] Launch community instance
- [ ] Implement reputation system
- [ ] Add collaborative filtering
- [ ] Create mobile app
- [ ] Build API ecosystem
- [ ] Publish benchmark datasets

## Success Metrics

### Scientific Impact
- Number of papers in knowledge graph
- Cross-domain insights discovered
- Citations of C-LIGHT in research
- Replication of C-LIGHT findings

### Community Health
- Active contributors
- Paper submission rate
- Weight adjustment proposals
- Code contributions

### Technical Performance
- Query latency (<500ms)
- Extraction accuracy (>85%)
- Graph consistency score
- Uptime (>99.5%)

## License & Governance

**Proposed License**: Apache 2.0 or MIT
- Allows commercial use
- Enables CANDLE integration
- Encourages community forks

**Governance Model**:
- Core team for architectural decisions
- Community voting for major changes
- Domain expert councils for quality control
- Transparent decision-making process

## Security & Privacy

### Open-Source Considerations
- No API keys in code (use environment variables)
- No proprietary algorithms
- No references to classified systems
- Sanitize file paths
- Remove internal hostnames/IPs

### Data Privacy
- Only public papers (no classified documents)
- Anonymize community contributions
- GDPR-compliant user data handling
- Transparent data usage policies

## Getting Started (For Contributors)

```bash
# Clone repository
git clone https://github.com/your-org/c-light.git

# Install dependencies
cd c-light
pip install -e ".[dev]"

# Run initial harvest
python -m c_light.harvest --categories cs.AI q-bio.NC --days 30

# Build knowledge graph
python -m c_light.build_graph

# Start RAG server
python -m c_light.serve
```

## Integration with CANDLE

```python
# Example: Using C-LIGHT as a knowledge source for CANDLE

from c_light import CLightKnowledgeGraph, CANDLEAdapter

# Initialize C-LIGHT
kg = CLightKnowledgeGraph()
kg.load_from_disk()

# For users with CANDLE access
if CANDLE_AVAILABLE:
    from candle import CognitiveScienceRAG

    candle_rag = CognitiveScienceRAG()
    adapter = CANDLEAdapter(
        clight_kg=kg,
        candle_rag=candle_rag
    )

    # Sync C-LIGHT insights to CANDLE
    adapter.sync_to_candle()

    # Use CANDLE's proprietary explainatory reasoning
    result = candle_rag.explain_behavioral_pattern(...)
```

---

**C-LIGHT**: Illuminating the path from scientific literature to actionable insights.
