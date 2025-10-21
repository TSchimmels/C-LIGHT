# C-LIGHT: Cognitive, Life-science, Intelligence Gathering & Hypothesis Testing

An open-source RAG system for identifying and analyzing quantum-level interactions in biological subjects, including **molecular interactions, drug interactions, epigenetics, brain organoids, neuronal excitability, metabolic cognition, neurotransmitter systems, environmental factors, and all physical interactions affecting cellular and organismal states**.

## 🎯 Multi-Scale Interaction Analysis

C-LIGHT traces quantum and molecular interactions across biological scales in any subject:

**Quantum → Molecular → Cellular → Tissue → Organism → Environmental**

- **Quantum Level**: Quantum coherence, electron tunneling, electromagnetic field interactions
- **Molecular Level**: Protein interactions, drug effects, epigenetic modifications, gene expression
- **Metabolic Level**: Nutrition, energy metabolism, neurotransmitter synthesis, cellular respiration
- **Cellular Level**: Firing thresholds, ion channels, membrane potentials, signal transduction
- **Tissue Level**: Organoid development, neural circuits, tissue organization, cellular networks
- **Organism Level**: Physiological states, metabolic profiles, response patterns
- **Environmental Level**: External factors, field interactions, substance exposure, physical conditions

### Research Applications

Understanding **interaction cascades** enables:
- **Quantum biology research**: How quantum effects manifest in biological systems
- **Drug interaction studies**: Tracing molecular effects through biological pathways
- **Organoid modeling**: Understanding development from molecular to tissue level
- **Metabolic research**: Mapping how nutritional states affect cellular function
- **Environmental interactions**: Identifying how external factors influence organism states
- **Cross-scale analysis**: Connecting molecular mechanisms to observable phenotypes

C-LIGHT reveals these interaction pathways for scientific research and discovery.

The C-LIGHT system provides two implementation strategies for managing academic papers across multiple scientific domains:

## Research Domains Covered
- **Cognitive Science**: Neuroscience, Psychology, Consciousness Studies
- **Life Sciences**: Nutrition, Microbiome, Exercise Physiology, Sleep Science
- **Molecular Interactions**: Protein-protein interactions, receptor binding, enzyme kinetics, signaling cascades, molecular mechanisms
- **Drug Interactions**: Pharmacokinetics, pharmacodynamics, drug-drug interactions, polypharmacy, drug-nutrient interactions
- **Temporal Pharmacology**: Absorption/onset, peak effects, half-life, clearance, crash/withdrawal dynamics, chronopharmacology
- **Substance Timing Dynamics**: Effects relative to food intake, circadian timing, interaction timing between multiple substances
- **Epigenetics**: DNA methylation, histone modification, gene expression regulation, environmental influences on gene activity
- **Neuronal Excitability**: Firing thresholds, action potentials, membrane potential, ion channels, synaptic transmission
- **Metabolic Cognition**: Fasting states, glucose/ketone metabolism, nutritional neuroscience, metabolic effects on brain function and neuronal firing
- **Brain Organoids**: 3D neural tissue models, developmental neuroscience, disease modeling, neural organogenesis, organoid electrophysiology, firing patterns
- **Tissue Engineering**: Organoid technology, bioengineering, stem cell biology, organ-on-chip systems
- **Nutritional Neuroscience**: Dietary effects on cognition, sugar metabolism, fasting, ketosis, micronutrients, brain energy metabolism, nutrient effects on excitability
- **Neuromodulation**: Factors affecting neuronal threshold (nutrition, metabolic state, ions, neurotransmitters, drugs, EMF)
- **Dopamine Systems**: Reward pathways, neurotransmitter circuits, dopamine regulation, neurochemical signaling
- **Behavioral Neuroscience**: Physiological factors affecting organism state, response patterns, neural development
- **Digital Environment Studies**: Screen exposure effects, electromagnetic field interactions, light spectrum influence
- **Social Biology**: Group behaviors, population dynamics, interaction patterns, environmental adaptations
- **Cultural Biology**: Environmental variations, population differences, developmental patterns
- **Cognitive Development**: Neural maturation, information processing, pattern recognition, learning mechanisms
- **Quantum Biology**: Quantum coherence in biological systems, quantum effects in cognition
- **EMF Biology**: Electromagnetic field effects on biological systems, cognition, and neuronal firing thresholds
- **Physics**: Quantum Physics (quant-ph), Biophysics (physics.bio-ph)
- **Biological Sciences**: Population biology, Ethology, Comparative biology
- **Pharmacology**: Drug effects on cellular function, organoid-based drug testing, compound interactions, receptor modulation

---

## Version 1: Real-time Processing (v1_realtime/)
**Use Case**: Continuous operation with GPU server always running

### Architecture
```
ArXiv API → Download to NVMe → Process Immediately → Train Models → Archive to HDD
```

### Components
- `arxiv_harvester.py`: Downloads papers directly to NVMe
- `batch_processor.py`: Processes papers immediately after download
- `doi_database.py`: Tracks all papers to prevent duplicates
- `paper_recall.py`: Retrieves archived papers when needed
- `doi_monitor.py`: Real-time monitoring dashboard

### Advantages
- Low latency from discovery to processing
- Continuous model updates
- Real-time insights

### Disadvantages
- High energy consumption (GPU server always on)
- Higher operational costs
- Not suitable for massive paper collections

### Usage
```bash
cd v1_realtime
# Cognitive Science & AI
python -m batch_processor --categories cs.AI cs.LG --continuous

# Quantum & Consciousness Research
python -m batch_processor --categories quant-ph q-bio.NC physics.bio-ph --continuous
```

## Version 2: Staged Processing (v2_staged/)
**Use Case**: Energy-efficient bulk processing of millions of papers

### Architecture
```
Stage 1 (Low-power server): ArXiv API → Download to HDD → Track in DOI DB
Stage 2 (GPU server): Move to NVMe → Batch Process → Train Models → Archive
```

### Components
- `hdd_harvester.py`: Downloads papers to HDD on low-power server
- `staged_batch_processor.py`: Moves papers to NVMe and processes in batches
- `doi_database.py`: Shared component for tracking papers
- `paper_recall.py`: Shared component for paper retrieval
- `doi_monitor.py`: Shared monitoring dashboard

### Advantages
- Energy efficient (GPU only runs when needed)
- Cost effective for large collections
- Can download millions of papers without GPU
- Flexible processing schedule

### Disadvantages
- Higher latency from discovery to processing
- Requires manual batch triggering
- More complex workflow

### Usage

**Stage 1: Continuous harvesting (low-power server)**
```bash
cd v2_staged
python hdd_harvester.py --continuous
```

**Stage 2: Batch processing (GPU server)**
```bash
cd v2_staged
python staged_batch_processor.py --category cs.AI --batches 10
```

## Choosing Between Versions

### Use Version 1 when:
- You have a dedicated GPU server
- Need real-time processing
- Paper volume is moderate (< 10K/day)
- Low latency is critical

### Use Version 2 when:
- Want to minimize energy costs
- Processing millions of papers
- Can batch process weekly/monthly
- Building large training datasets

## Storage Architecture

Both versions use the same storage strategy:

### NVMe Storage (Fast)
- DOI database for deduplication
- Active processing workspace
- LRU cache for frequently accessed papers
- Model checkpoints

### HDD Storage (Large)
- Raw downloaded papers (v2)
- Processed paper archive
- Historical data
- Backup and recovery

## DOI Database Features

The DOI database is the core component shared by both versions:

- **Deduplication**: Prevents downloading/processing the same paper twice
- **Tracking**: Complete lifecycle tracking from download to archive
- **Search**: Fast queries by DOI, ArXiv ID, category, date range
- **Metadata**: Stores embeddings, causal relations, quality scores
- **Integration**: Works with both processing strategies

## Monitoring

Both versions include comprehensive monitoring:

```bash
# Web dashboard
python -c "from doi_monitor import create_web_dashboard; app = create_web_dashboard(monitor); app.run()"

# CLI status
python doi_monitor.py --status
```

## Migration Between Versions

You can migrate from v1 to v2 or vice versa:

```bash
# Export from v1
cd v1_realtime
python doi_database.py --export /tmp/papers.json

# Import to v2
cd v2_staged
python doi_database.py --import /tmp/papers.json
```

## Hardware Requirements

### Version 1 (Real-time)
- **GPU Server**: 8x V100 32GB or better
- **NVMe**: 2-4TB for processing
- **HDD**: 32TB+ for archive
- **Network**: 1Gbps+ for continuous downloads

### Version 2 (Staged)
- **Harvest Server**: Low-power CPU (Raspberry Pi 4+ works)
- **HDD**: 32TB+ for raw storage
- **GPU Server**: 8x V100 32GB (only when processing)
- **NVMe**: 2-4TB for batch processing

## Best Practices

1. **Start with v2** for initial bulk collection
2. **Switch to v1** for ongoing real-time updates
3. **Monitor storage** regularly
4. **Schedule processing** during off-peak hours
5. **Backup DOI database** weekly

## Example Workflows

### Research Project (v2)
```bash
# 1. Define categories of interest
echo '["cs.AI", "cs.LG", "stat.ML", "quant-ph", "q-bio.NC", "physics.bio-ph"]' > categories.json

# 2. Harvest papers for 1 month
python hdd_harvester.py --continuous

# 3. Process in batches weekly
python staged_batch_processor.py --batches 50

# 4. Train models on processed data
python train_models.py --checkpoint latest
```

### Production System (v1)
```bash
# 1. Start continuous processing
python batch_processor.py --continuous &

# 2. Monitor system
python doi_monitor.py --web

# 3. Set up alerts
python doi_monitor.py --alert-email admin@example.com
```

## Performance Metrics

### Version 1
- Download rate: 10-50 papers/minute
- Processing rate: 5-20 papers/minute
- Model update frequency: Every 1000 papers
- Energy usage: ~2kW continuous

### Version 2
- Download rate: 100-500 papers/minute (no processing)
- Batch processing: 10K papers/hour
- Model update frequency: Per batch
- Energy usage: ~100W (harvest) + 2kW (processing)

## ArXiv Categories Reference

### Cognitive Science & AI
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.HC` - Human-Computer Interaction
- `stat.ML` - Machine Learning (Statistics)

### Quantum Physics & Biology
- `quant-ph` - Quantum Physics (quantum mechanics, quantum information, quantum computing)
- `q-bio.NC` - Neurons and Cognition (computational neuroscience, consciousness studies)
- `physics.bio-ph` - Biological Physics (quantum biology, biophysics, EMF effects)

### Life Sciences
- `q-bio.MN` - Molecular Networks (systems biology, metabolic networks)
- `q-bio.BM` - Biomolecules (protein structure, molecular dynamics)
- `q-bio.TO` - Tissues and Organs (physiology, organ systems, organoid development)
- `q-bio.CB` - Cell Behavior (stem cells, neural differentiation, organoid culture)

### Brain Organoid & Tissue Engineering
- `q-bio.TO` - Tissues and Organs (brain organoids, cerebral organoids, neural tissue engineering)
- `q-bio.CB` - Cell Behavior (neural stem cells, organoid assembly, 3D cell culture)
- `q-bio.SC` - Subcellular Processes (cellular differentiation, morphogenesis)
- `q-bio.NC` - Neurons and Cognition (organoid electrophysiology, neural network development)
- `physics.bio-ph` - Biological Physics (biomechanics of organoid development, biophysical modeling)

### Social Sciences & Digital Influence
- `cs.CY` - Computers and Society (social media, digital manipulation, online behavior)
- `cs.HC` - Human-Computer Interaction (persuasive design, dark patterns, UX manipulation)
- `cs.SI` - Social and Information Networks (social network analysis, information spread)
- `physics.soc-ph` - Physics and Society (social dynamics, cultural evolution, collective behavior)
- `econ.GN` - General Economics (behavioral economics, decision-making)

### Additional Resources
- **PubMed**: Biomedical literature, EMF studies, consciousness research, organoid research, drug interactions, epigenetics
- **bioRxiv**: Preprints in biology, neuroscience, organoid development, tissue engineering, molecular biology
- **PsyArXiv**: Psychology preprints (social psychology, political psychology, behavioral science)
- **SocArXiv**: Social science preprints (sociology, cultural studies, social media research)
- **PhilSci Archive**: Philosophy of mind, consciousness theories
- **OSF Preprints**: Open science, replication studies, consciousness research
- **Cell Press**: Stem Cell Reports, Cell Stem Cell (organoid & stem cell research), Molecular Cell, Cell Metabolism
- **Nature**: Nature Protocols, Nature Methods (organoid methodologies), Nature Neuroscience, Nature Human Behaviour
- **SSRN**: Social Science Research Network (political psychology, behavioral economics)
- **DrugBank**: Drug interaction databases, pharmacology data

## Example Research Queries

### Quantum Biology & Consciousness
```bash
# Harvest quantum consciousness papers
python arxiv_harvester.py --categories quant-ph q-bio.NC --keywords "quantum consciousness,microtubules,Orch-OR"

# EMF biology research
python arxiv_harvester.py --categories physics.bio-ph --keywords "electromagnetic fields,EMF,cognition,neural"
```

### Cross-Domain Research
```bash
# Multi-domain harvest for comprehensive research
python hdd_harvester.py --categories cs.AI quant-ph q-bio.NC physics.bio-ph --continuous
```

### Metabolic & Nutritional Cognition Research
```bash
# Fasting and metabolic states
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "fasting,intermittent fasting,ketosis,glucose metabolism,brain energy,metabolic switching"

# Sugar, glucose, and cognitive performance
python hdd_harvester.py --categories q-bio.NC q-bio.MN physics.bio-ph --keywords "glucose,sugar,insulin,cognitive performance,glycemic,blood sugar,hypoglycemia"

# Nutritional effects on brain function
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "nutrition,diet,cognitive function,brain health,micronutrients,omega-3,ketogenic,Mediterranean diet"

# Metabolic effects in brain organoids
python hdd_harvester.py --categories q-bio.TO q-bio.MN --keywords "organoid,metabolism,glucose,ketones,energy metabolism,mitochondrial,metabolic state"
```

### Brain Organoid Research
```bash
# Harvest brain organoid and tissue engineering papers
python arxiv_harvester.py --categories q-bio.TO q-bio.CB q-bio.NC --keywords "brain organoid,cerebral organoid,neural organoid,organoid culture,cortical organoid"

# Organoid development and maturation
python arxiv_harvester.py --categories q-bio.TO q-bio.CB physics.bio-ph --keywords "organoid development,tissue engineering,3D culture,stem cell differentiation,neural maturation,organogenesis"

# Disease modeling with organoids (broad scope)
python hdd_harvester.py --categories q-bio.TO q-bio.NC --keywords "disease modeling,organoid,neurodevelopmental,neurodegeneration,Alzheimer,autism,schizophrenia,epilepsy,brain disease"

# Organoid-based research (not just pharma)
python arxiv_harvester.py --categories q-bio.TO q-bio.QM q-bio.NC --keywords "organoid,neural development,brain development,developmental biology,neurobiology,synaptic,neural circuits"

# Nutritional and metabolic effects on organoids
python hdd_harvester.py --categories q-bio.TO q-bio.MN --keywords "organoid,nutrition,metabolic,glucose,fasting,ketones,metabolic state,energy metabolism"
```

### Integrative Brain Organoid Studies
```bash
# Cross-domain organoid research: development, cognition, and consciousness
python hdd_harvester.py --categories q-bio.TO q-bio.NC q-bio.CB q-bio.MN quant-ph physics.bio-ph --keywords "organoid,neural,consciousness,quantum,cognition,metabolism" --continuous

# Organoid electrophysiology and neural networks
python arxiv_harvester.py --categories q-bio.NC physics.bio-ph --keywords "organoid electrophysiology,neural activity,brain waves,MEA,multielectrode array,neural oscillations,spontaneous activity"

# Organoid-based studies of metabolic cognition
python arxiv_harvester.py --categories q-bio.TO q-bio.NC q-bio.MN --keywords "organoid,metabolic state,glucose,ketones,fasting,neural function,synaptic plasticity"

# Comprehensive brain organoid biology
python hdd_harvester.py --categories q-bio.TO q-bio.CB q-bio.NC q-bio.MN --keywords "brain organoid,neural organoid,organoid maturation,cell types,glia,neurons,vascularization,organoid complexity" --continuous
```

### Neuronal Firing Threshold & Excitability Research
```bash
# Neuronal excitability and firing thresholds
python arxiv_harvester.py --categories q-bio.NC physics.bio-ph --keywords "firing threshold,action potential,membrane potential,neuronal excitability,spike threshold,excitability"

# Ion channels and threshold modulation
python hdd_harvester.py --categories q-bio.NC q-bio.BM --keywords "ion channels,sodium channels,potassium channels,calcium channels,voltage-gated,threshold modulation"

# Metabolic effects on neuronal firing
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "glucose,ketones,metabolism,firing rate,neuronal activity,energy metabolism,ATP,mitochondria,excitability"

# Nutritional effects on neuronal excitability
python hdd_harvester.py --categories q-bio.NC q-bio.MN --keywords "nutrition,diet,neuronal excitability,firing threshold,electrolytes,ions,magnesium,calcium,sodium,potassium"

# Fasting effects on neuronal firing
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "fasting,caloric restriction,neuronal activity,firing patterns,brain waves,EEG,metabolic state,excitability"

# Brain organoid firing patterns and threshold manipulation
python hdd_harvester.py --categories q-bio.TO q-bio.NC --keywords "organoid,electrophysiology,firing,action potential,MEA,multielectrode,neural activity,spontaneous activity,threshold"

# Neuromodulation and threshold changes
python arxiv_harvester.py --categories q-bio.NC physics.bio-ph --keywords "neuromodulation,threshold,excitability,synaptic,neurotransmitters,modulation,plasticity,homeostasis"

# EMF effects on neuronal firing thresholds
python arxiv_harvester.py --categories physics.bio-ph q-bio.NC --keywords "electromagnetic,EMF,neuronal firing,excitability,threshold,magnetic field,electric field,TMS,tDCS"
```

### Integrative Threshold Modulation Studies
```bash
# Comprehensive threshold manipulation research
python hdd_harvester.py --categories q-bio.NC q-bio.MN physics.bio-ph q-bio.TO --keywords "firing threshold,excitability,modulation,metabolism,nutrition,glucose,ketones,ions,channels" --continuous

# Cross-domain: metabolism, nutrition, and neural excitability
python arxiv_harvester.py --categories q-bio.NC q-bio.MN q-bio.TO --keywords "metabolic state,nutritional,neuronal firing,threshold,excitability,energy,ATP,organoid,electrophysiology"

# Sugar/glucose effects on neuronal activity
python hdd_harvester.py --categories q-bio.NC q-bio.MN --keywords "glucose,sugar,insulin,neuronal activity,firing rate,excitability,energy availability,glycolysis,brain energy"
```

### Psychology & Human Development Research
```bash
# Personality psychology across the lifespan
python arxiv_harvester.py --categories cs.CY physics.soc-ph --keywords "personality,Big Five,MBTI,traits,temperament,character,personality development"

# Developmental psychology and maturation
python hdd_harvester.py --categories cs.CY physics.soc-ph --keywords "developmental stages,maturation,psychosocial,Erikson,life stages,adolescence,adulthood,aging,lifespan"

# Humanistic psychology and self-actualization
python arxiv_harvester.py --categories cs.CY --keywords "Maslow,hierarchy of needs,self-actualization,human potential,motivation,growth,self-determination"

# Cognitive development and personality
python hdd_harvester.py --categories cs.CY q-bio.NC --keywords "cognitive development,personality,individual differences,cognitive styles,traits,intelligence,development"
```

### Molecular Interactions & Drug Interactions
```bash
# Protein-protein interactions and signaling cascades
python arxiv_harvester.py --categories q-bio.BM q-bio.MN --keywords "protein interaction,receptor binding,signaling pathway,molecular mechanism,enzyme kinetics,binding affinity"

# Drug interactions and pharmacology
python hdd_harvester.py --categories q-bio.QM q-bio.BM --keywords "drug interaction,pharmacokinetics,pharmacodynamics,polypharmacy,drug-drug interaction,adverse effects"

# Drug-nutrient interactions
python arxiv_harvester.py --categories q-bio.BM q-bio.MN --keywords "drug-nutrient interaction,food-drug interaction,metabolism,bioavailability,absorption"

# Neurotransmitter systems and receptor interactions
python hdd_harvester.py --categories q-bio.NC q-bio.BM --keywords "neurotransmitter,receptor,dopamine,serotonin,GABA,glutamate,receptor binding,signaling"
```

### Epigenetics & Gene-Environment Interactions
```bash
# Epigenetic modifications and gene regulation
python arxiv_harvester.py --categories q-bio.GN q-bio.MN --keywords "epigenetics,DNA methylation,histone modification,gene expression,chromatin,epigenome"

# Environmental influences on gene expression
python hdd_harvester.py --categories q-bio.GN q-bio.NC --keywords "gene-environment interaction,epigenetic regulation,stress,trauma,early life,developmental programming"

# Epigenetics of behavior and emotion
python arxiv_harvester.py --categories q-bio.GN q-bio.NC --keywords "behavioral epigenetics,emotional regulation,temperament,personality,epigenetic inheritance,transgenerational"

# Nutritional epigenetics
python hdd_harvester.py --categories q-bio.GN q-bio.MN --keywords "nutritional epigenetics,diet,fasting,methyl donors,folate,epigenetic modulation"
```

### Dopamine Systems & Reward Pathways
```bash
# Dopamine neuroscience and reward circuits
python arxiv_harvester.py --categories q-bio.NC --keywords "dopamine,reward pathway,ventral tegmental area,nucleus accumbens,mesolimbic,motivation,incentive salience"

# Addiction and dopamine dysregulation
python hdd_harvester.py --categories q-bio.NC physics.bio-ph --keywords "addiction,substance abuse,dopamine,reward,craving,withdrawal,relapse,sensitization"

# Social media and dopamine manipulation
python arxiv_harvester.py --categories cs.CY cs.HC q-bio.NC --keywords "social media,dopamine,reward,likes,notifications,engagement,variable reward,addiction"

# Dopamine and decision-making
python hdd_harvester.py --categories q-bio.NC cs.AI --keywords "dopamine,decision making,reinforcement learning,value,prediction error,expected reward"
```

### Digital Environment & Screen Exposure
```bash
# Screen exposure and light effects
python arxiv_harvester.py --categories cs.CY cs.HC q-bio.NC --keywords "screen exposure,blue light,circadian rhythm,light exposure,display technology,visual stimulation"

# Digital interface design
python hdd_harvester.py --categories cs.CY cs.SI cs.HC --keywords "user interface,interaction design,digital environment,engagement patterns,attention patterns,screen time"

# Technology usage patterns
python arxiv_harvester.py --categories cs.CY cs.HC q-bio.NC --keywords "technology use,screen time,digital exposure,usage patterns,interaction frequency,device interaction"

# Information processing
python hdd_harvester.py --categories cs.CY cs.SI physics.soc-ph --keywords "information processing,pattern recognition,data presentation,visual information,cognitive load,information exposure"
```

### Social Biology & Group Dynamics
```bash
# Group interaction patterns
python arxiv_harvester.py --categories physics.soc-ph q-bio.PE --keywords "group behavior,collective dynamics,population patterns,social organization,group structure,interaction networks"

# Population dynamics
python hdd_harvester.py --categories physics.soc-ph q-bio.PE --keywords "population biology,group size,density effects,proximity patterns,spacing behavior,aggregation patterns"

# Collective behavior
python arxiv_harvester.py --categories physics.soc-ph q-bio.NC --keywords "collective movement,group coordination,synchronized behavior,swarm intelligence,emergent patterns"

# Environmental variations
python hdd_harvester.py --categories physics.soc-ph q-bio.PE --keywords "environmental adaptation,habitat differences,geographic variation,population differences,ecological factors"
```

### Cognitive Development & Pattern Formation
```bash
# Pattern recognition development
python arxiv_harvester.py --categories cs.CY cs.AI q-bio.NC --keywords "pattern recognition,cognitive development,learning patterns,information organization,schema formation,mental models"

# Information processing patterns
python hdd_harvester.py --categories cs.CY q-bio.NC --keywords "information processing,cognitive patterns,learning mechanisms,memory formation,pattern storage,neural encoding"

# Decision-making mechanisms
python hdd_harvester.py --categories cs.CY q-bio.NC cs.AI --keywords "decision processes,choice mechanisms,preference formation,valuation systems,option evaluation"

# Value systems and preferences
python arxiv_harvester.py --categories q-bio.NC physics.soc-ph --keywords "preference formation,value encoding,reward processing,priority systems,evaluation mechanisms"
```

### Population Neuroscience & Environmental Variations
```bash
# Population neurobiology
python arxiv_harvester.py --categories q-bio.NC physics.soc-ph --keywords "population neuroscience,geographic variation,environmental effects,population differences,comparative neuroscience"

# Environmental adaptation patterns
python hdd_harvester.py --categories physics.soc-ph q-bio.PE --keywords "environmental adaptation,developmental plasticity,phenotypic variation,environmental influences,adaptive responses"

# Population-level patterns
python arxiv_harvester.py --categories physics.soc-ph q-bio.NC --keywords "population patterns,group-level traits,demographic variation,environmental factors,geographic differences"
```

### Multi-Scale Interaction Chains
```bash
# Comprehensive cross-scale interaction research
python hdd_harvester.py \
  --categories q-bio.NC q-bio.MN q-bio.BM q-bio.GN quant-ph physics.bio-ph \
  --keywords "epigenetics,gene expression,metabolism,neurotransmitter,quantum effects,molecular interactions,environment,development,cellular function,organism state" \
  --continuous

# Molecular to cellular cascades
python arxiv_harvester.py --categories q-bio.BM q-bio.NC --keywords "molecular mechanism,signaling cascade,cellular response,neurotransmitter,receptor,pathway,physiological outcome"
```

### Temporal Pharmacology & Substance Timing Dynamics
```bash
# Pharmacokinetics and temporal dynamics
python arxiv_harvester.py --categories q-bio.QM q-bio.BM --keywords "pharmacokinetics,absorption,peak,half-life,clearance,bioavailability,time course,temporal dynamics"

# Caffeine timing and interactions
python hdd_harvester.py --categories q-bio.NC q-bio.MN q-bio.BM --keywords "caffeine,absorption,peak,half-life,adenosine,food interaction,timing,tolerance,withdrawal"

# Alcohol pharmacodynamics and timing
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "alcohol,ethanol,pharmacokinetics,absorption,peak,blood alcohol,metabolism,ADH,ALDH,food effects,hangover"

# Sugar/glucose dynamics and timing
python hdd_harvester.py --categories q-bio.MN q-bio.NC --keywords "glucose,glycemic index,insulin response,blood sugar,glucose spike,crash,hypoglycemia,timing,food combination"

# Drug timing relative to food
python arxiv_harvester.py --categories q-bio.QM q-bio.BM --keywords "food-drug interaction,timing,absorption,bioavailability,fed state,fasted state,meal timing"

# Multi-substance interaction timing
python hdd_harvester.py --categories q-bio.QM q-bio.NC q-bio.BM --keywords "drug interaction,timing,sequence,combination,synergy,antagonism,pharmacokinetic interaction"
```

### Common Substance Temporal Profiles
```bash
# Caffeine comprehensive timing
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "caffeine,onset,peak,duration,half-life,tolerance,dependence,withdrawal,adenosine receptor,cortisol,sleep,circadian"

# Alcohol stages and effects
python hdd_harvester.py --categories q-bio.NC q-bio.MN --keywords "alcohol,intoxication,peak,metabolism,acetaldehyde,GABA,glutamate,dopamine,hangover,recovery,withdrawal"

# Sugar/carbohydrate timing dynamics
python arxiv_harvester.py --categories q-bio.MN q-bio.NC --keywords "sugar,glucose,insulin,glycemic,postprandial,glucose curve,insulin spike,reactive hypoglycemia,timing,GLP-1"

# Nicotine timing and dopamine
python hdd_harvester.py --categories q-bio.NC q-bio.BM --keywords "nicotine,absorption,peak,half-life,dopamine,acetylcholine,withdrawal,craving,timing,tolerance"

# Exercise timing effects
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "exercise,timing,acute,BDNF,dopamine,endorphins,cortisol,post-exercise,recovery,circadian,fasted exercise"
```

### Chronopharmacology & Circadian Timing
```bash
# Circadian effects on drug metabolism
python arxiv_harvester.py --categories q-bio.QM q-bio.NC --keywords "chronopharmacology,circadian,drug metabolism,CYP450,timing,morning,evening,diurnal,time of day"

# Meal timing and metabolic effects
python hdd_harvester.py --categories q-bio.MN q-bio.NC --keywords "meal timing,time-restricted eating,circadian,metabolic,insulin sensitivity,glucose,cortisol,melatonin"

# Sleep-substance interactions
python arxiv_harvester.py --categories q-bio.NC --keywords "sleep,caffeine,alcohol,timing,sleep quality,REM,adenosine,melatonin,circadian disruption"
```

### Combination Effects & Interaction Timing
```bash
# Caffeine + sugar interactions
python arxiv_harvester.py --categories q-bio.NC q-bio.MN --keywords "caffeine,sugar,glucose,insulin,dopamine,energy,crash,combination,synergy"

# Caffeine + L-theanine timing
python hdd_harvester.py --categories q-bio.NC q-bio.BM --keywords "caffeine,L-theanine,timing,synergy,attention,relaxation,alpha waves,GABA"

# Alcohol + caffeine interactions
python arxiv_harvester.py --categories q-bio.NC q-bio.QM --keywords "alcohol,caffeine,interaction,masking,intoxication,risk,energy drink,timing"

# Protein + carbohydrate timing
python arxiv_harvester.py --categories q-bio.MN --keywords "protein,carbohydrate,timing,insulin,amino acids,glycemic,meal composition,sequence"

# Drug + nutrient depletion over time
python hdd_harvester.py --categories q-bio.QM q-bio.MN --keywords "drug-induced,nutrient depletion,chronic,timing,cumulative,B vitamins,magnesium,CoQ10"
```

### Substance Onset, Peak, and Crash Profiles
```bash
# Comprehensive temporal profiles of common substances
python hdd_harvester.py \
  --categories q-bio.NC q-bio.MN q-bio.QM q-bio.BM \
  --keywords "pharmacokinetics,onset,peak,duration,half-life,elimination,crash,withdrawal,rebound,tolerance,sensitization,caffeine,alcohol,sugar,glucose,nicotine" \
  --continuous

# Withdrawal and crash dynamics
python arxiv_harvester.py --categories q-bio.NC q-bio.BM --keywords "withdrawal,crash,rebound,abstinence,symptoms,timeline,acute,protracted,PAWS,dopamine,receptor"

# Tolerance development timing
python hdd_harvester.py --categories q-bio.NC q-bio.BM --keywords "tolerance,sensitization,receptor downregulation,upregulation,chronic,adaptation,time course,development"
```

## Comprehensive Multi-Domain Research Workflow

### Complete C-LIGHT Research Pipeline
For researchers interested in the full scope of cognitive science, brain function, and human development:

```bash
# Step 1: Start comprehensive harvesting (all domains including upstream factors)
python hdd_harvester.py \
  --categories cs.AI cs.HC cs.CY cs.SI q-bio.NC q-bio.TO q-bio.CB q-bio.MN q-bio.BM q-bio.GN quant-ph physics.bio-ph physics.soc-ph econ.GN \
  --keywords "cognition,brain,organoid,metabolism,fasting,glucose,ketones,firing,threshold,excitability,personality,development,consciousness,quantum,epigenetics,dopamine,social media,hypernudging,culture,belief,drug interaction,molecular" \
  --continuous

# Step 2: Process in batches (GPU server)
python staged_batch_processor.py --batches 100

# Step 3: Query cross-domain insights - UPSTREAM TO DOWNSTREAM CAUSALITY
# Examples:
# - "How does fasting affect neuronal firing thresholds in brain organoids?"
# - "What epigenetic changes influence temperament and personality development?"
# - "How do social media algorithms manipulate dopamine systems to drive behavior?"
# - "What molecular interactions and drug interactions affect cognitive performance?"
# - "How do cultural norms influence gene expression and brain development?"
# - "What upstream factors (molecular → metabolic → neural → behavioral → social) drive political beliefs?"
```

## Key Research Questions C-LIGHT Can Answer

### Temporal Pharmacology & Substance Timing
- What is the complete time course (onset → peak → crash) for caffeine, alcohol, sugar, and other common substances?
- How does food timing affect drug absorption, peak effects, and duration?
- What are the optimal timing windows for combining substances (caffeine + L-theanine, protein + carbs)?
- How do circadian rhythms affect drug metabolism and efficacy?
- What is the timeline of tolerance development and withdrawal for different substances?
- How do substance crashes affect cognition, mood, and decision-making?
- What interaction timing creates synergistic vs. antagonistic effects?
- How does chronic substance use alter temporal response profiles?

### Specific Substance Temporal Profiles
- **Caffeine**: Onset (15-45min), Peak (1-2hr), Half-life (3-7hr), Crash timing, Sleep effects, Tolerance timeline
- **Alcohol**: Absorption rate (food effects), Peak BAC timing, Metabolism rate, Hangover duration, Withdrawal timeline
- **Sugar/Glucose**: Glycemic response curve, Insulin spike timing, Reactive hypoglycemia (2-4hr post), Energy crash
- **Nicotine**: Rapid onset (7-10sec inhaled), Peak (10min), Short half-life (2hr), Craving cycles, Withdrawal phases
- **Exercise**: Acute BDNF release, Endorphin timeline, Cortisol response, Post-exercise recovery, Adaptation timeline

### Upstream Molecular & Epigenetic Factors
- How do epigenetic modifications influence behavior, emotion, and temperament?
- What molecular interactions and signaling cascades lead to personality changes?
- How do environmental factors trigger epigenetic changes that affect cognition?
- What drug interactions influence neurotransmitter systems and behavior?
- How does nutrition affect gene expression and brain development?

### Dopamine Systems & Neural Signaling
- How do dopamine pathways regulate cellular signaling, reward processing, and neural function?
- What factors affect dopamine release (drugs, nutrition, environmental stimuli)?
- How does dopamine dysregulation affect neural circuits and cellular states?
- What is the relationship between dopamine signaling and neural development?
- How can understanding dopamine systems inform therapeutic approaches?

### Digital Environment & Sensory Exposure
- How do digital displays affect circadian rhythms and neural function?
- What are the mechanisms of light exposure on cellular signaling?
- How do visual stimuli patterns affect attention and neural processing?
- What physiological responses occur from different screen exposures?
- How does exposure timing affect biological rhythms?
- What are the effects of information density on cognitive load?

### Group Biology & Population Dynamics
- How do group proximities affect individual organism states?
- What drives aggregation and dispersal dynamics in populations?
- How does group size affect individual physiology?
- What role does spatial organization play in population health?
- How do different environmental contexts shape group behaviors?

### Neural Development & Pattern Formation
- How are neural patterns formed and maintained?
- What causes variation in neural development across individuals?
- How do information processing mechanisms develop?
- What factors drive individual differences in cognition?
- How do different neural architectures process information?
- What makes certain neural patterns stable over time?

### Population Neuroscience & Environmental Context
- How do environmental variations influence neural development?
- What neurobiological differences exist across populations?
- How are environmental adaptations encoded neurally?
- What role do ecological factors play in neural function?
- How do different environmental contexts affect brain structure?

### Neuronal Excitability & Metabolism
- How do glucose vs. ketone metabolism affect neuronal firing thresholds?
- What nutritional factors modulate ion channel function and excitability?
- How does fasting change brain wave patterns and neuronal synchronization?
- What metabolic states optimize cognitive performance?

### Brain Organoid Biology
- How do brain organoids model human neurodevelopment and disease?
- What firing patterns emerge in maturing brain organoids?
- How do metabolic conditions affect organoid development and function?
- Can organoids model consciousness or complex cognitive processes?
- How can organoids test drug interactions and molecular mechanisms?

### Developmental & Personality Psychology
- How do personality traits change across the lifespan?
- What factors influence maturation and self-actualization?
- How does developmental stage affect cognitive processing?
- What is the relationship between personality and brain function?

### Integrative Multi-Scale Interaction Questions (QUANTUM → ORGANISM)
- **Molecular → Cellular**: How do protein interactions → neurotransmitter release → membrane potentials → cellular states?
- **Epigenetic → Physiological**: How does environmental exposure → epigenetic changes → gene expression → organism phenotype?
- **Metabolic → Neural**: How does fasting → ketosis → neuronal firing → network activity → organism state?
- **Drug → Physiological**: How do drug interactions → receptor binding → cellular signaling → tissue function → organism response?
- **Environmental → Neural**: How does light exposure → circadian signaling → neural activity → sleep patterns → organism rhythms?
- **Quantum → Biological**: How do quantum effects → molecular tunneling → enzyme function → metabolic pathways → cellular energy?
- **Complete Chain**: How does nutrition → epigenetics → gene expression → protein synthesis → neurotransmitter function → neuronal firing → circuit activity → regional processing → organism state → behavioral output?

### Temporal Causal Chain Questions (WITH TIMING)
- **Caffeine Cascade (0-8hr)**: T0: Caffeine ingestion → T15min: Absorption begins → T45min: Adenosine receptor blockade → T1hr: Peak dopamine/norepinephrine → T2hr: Peak alertness → T3-5hr: Half-life elimination → T6-8hr: Adenosine rebound → Crash → Withdrawal symptoms → Sleep disruption (if late)
- **Alcohol Timeline (0-24hr)**: T0: Alcohol ingestion → T15-45min (food dependent): Peak BAC → GABA↑/Glutamate↓ → Dopamine release → Euphoria → T2-3hr: Metabolism begins (ADH→Acetaldehyde) → T4-12hr: Hangover (inflammation, dehydration, acetaldehyde toxicity, glutamate rebound) → Mood crash
- **Sugar Crash Cycle (0-4hr)**: T0: Sugar ingestion → T15min: Blood glucose spike → T30min: Insulin spike → T1hr: Peak insulin → T1.5-2hr: Rapid glucose clearance → T2-4hr: Reactive hypoglycemia → Energy crash → Mood drop → Cravings → Repeat cycle
- **Caffeine+Sugar Synergy**: Combined dopamine spike → Enhanced crash → Insulin resistance over time → Altered reward sensitivity → Craving cycles
- **Exercise-Fasted State**: Fasted state → Enhanced fat oxidation → Ketone production → BDNF release → Improved neuroplasticity (vs. fed state with glucose dominant)
- **Drug Timing + Food**: Empty stomach → Faster absorption → Higher peak → Shorter duration vs. With food → Slower absorption → Lower peak → Longer duration → Different side effect profile
- **Chronic Substance Timeline**: Week 1: Initial effects → Week 2-4: Tolerance develops (receptor downregulation) → Month 2-3: Baseline shift (new normal) → Cessation: Withdrawal (receptor upregulation lag) → Weeks-months: PAWS (protracted withdrawal) → Brain adaptation timeline