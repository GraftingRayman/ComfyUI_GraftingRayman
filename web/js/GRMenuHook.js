import { app } from "/scripts/app.js";

function getGroupUnderMouse(canvas) {
    const [x, y] = canvas.graph_mouse;
    return canvas.graph.getGroupOnPos?.(x, y) ?? null;
}

function getNodesInGroup(group) {
    return app.graph._nodes.filter(node => {
        if (!node.pos) return false;

        const w = node.size?.[0] || 200;
        const h = node.size?.[1] || 100;
        const cx = node.pos[0] + w / 2;
        const cy = node.pos[1] + h / 2;

        return (
            cx >= group._pos[0] &&
            cx <= group._pos[0] + group._size[0] &&
            cy >= group._pos[1] &&
            cy <= group._pos[1] + group._size[1]
        );
    });
}

function getUngroupedNodes() {
    const allNodes = app.graph._nodes;
    const groups = app.graph._groups || [];
    
    return allNodes.filter(node => {
        if (!node.pos) return false;
        
        const w = node.size?.[0] || 200;
        const h = node.size?.[1] || 100;
        const cx = node.pos[0] + w / 2;
        const cy = node.pos[1] + h / 2;
        
        const isInGroup = groups.some(group => {
            return (
                cx >= group._pos[0] &&
                cx <= group._pos[0] + group._size[0] &&
                cy >= group._pos[1] &&
                cy <= group._pos[1] + group._size[1]
            );
        });
        
        return !isInGroup;
    });
}

function createVirtualGroupForUngrouped(canvas, nodes) {
    if (nodes.length === 0) {
        return {
            _pos: [100, 100],
            _size: [800, 600],
            title: "Ungrouped Nodes"
        };
    }
    
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    
    nodes.forEach(node => {
        if (!node.pos) return;
        
        const w = node.size?.[0] || 200;
        const h = node.size?.[1] || 100;
        const x = node.pos[0];
        const y = node.pos[1];
        
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + w);
        maxY = Math.max(maxY, y + h);
    });
    
    const padding = 100;
    const groupX = minX - padding;
    const groupY = minY - padding;
    const groupWidth = (maxX - minX) + (padding * 2);
    const groupHeight = (maxY - minY) + (padding * 2);
    
    const minWidth = 800;
    const minHeight = 600;
    
    return {
        _pos: [Math.max(0, groupX), Math.max(0, groupY)],
        _size: [
            Math.max(minWidth, groupWidth),
            Math.max(minHeight, groupHeight)
        ],
        title: "Ungrouped Nodes"
    };
}

function calculateLayoutBounds(group) {
    const margin = 20; // Changed from 10 to be consistent
    const startX = group._pos[0] + margin;
    const startY = group._pos[1] + margin + 30;
    const availableWidth = group._size[0] - (2 * margin);
    const availableHeight = group._size[1] - (2 * margin) - 30;
    
    return {
        startX,
        startY,
        availableWidth,
        availableHeight,
        margin,
        groupRight: group._pos[0] + group._size[0] - margin,
        groupBottom: group._pos[1] + group._size[1] - margin,
        groupTopWithTitle: group._pos[1] + margin + 30
    };
}

function applyBoundaryCheck(node, bounds, nodeWidth, nodeHeight) {
    const [x, y] = node.pos;
    let adjustedX = x;
    let adjustedY = y;
    
    if (x < bounds.startX) {
        adjustedX = bounds.startX;
    }
    
    if (y < bounds.groupTopWithTitle) {
        adjustedY = bounds.groupTopWithTitle;
    }
    
    if (x + nodeWidth > bounds.groupRight) {
        adjustedX = bounds.groupRight - nodeWidth;
    }
    
    if (y + nodeHeight > bounds.groupBottom) {
        adjustedY = bounds.groupBottom - nodeHeight;
    }
    
    if (adjustedX !== x || adjustedY !== y) {
        console.log(`   ↪️ Adjusting position: [${x}, ${y}] → [${adjustedX}, ${adjustedY}]`);
        node.pos[0] = adjustedX;
        node.pos[1] = adjustedY;
    }
}

function getUniqueGroupName(baseName) {
    // Get all existing group titles
    const existingTitles = new Set();
    if (app.graph && app.graph._groups) {
        app.graph._groups.forEach(group => {
            if (group.title) {
                existingTitles.add(group.title);
            }
        });
    }
    
    // Check if base name already exists
    if (!existingTitles.has(baseName)) {
        return baseName;
    }
    
    // Add date and time suffix
    const now = new Date();
    const dateStr = now.toISOString()
        .replace(/T/, '_')
        .replace(/\..+/, '')
        .replace(/:/g, '-');
    
    let newName = `${baseName}_${dateStr}`;
    
    // If still exists, add a number
    let counter = 1;
    while (existingTitles.has(newName)) {
        newName = `${baseName}_${dateStr}_${counter}`;
        counter++;
    }
    
    return newName;
}

function importWorkflowIntoGroup(mousePos, targetGroup = null) {
    console.log('📥 Import workflow into group starting...');
    console.log('🎯 Target group:', targetGroup ? `"${targetGroup.title}"` : 'New group');
    
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";

    input.onchange = async () => {
        const file = input.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const data = JSON.parse(text);

            const nodes = data.nodes || [];
            const links = data.links || [];

            const nodeMap = {};
            
            // Determine import position
            let importOffsetX, importOffsetY;
            
            if (targetGroup) {
                // Import into existing group - center within the group
                importOffsetX = targetGroup._pos[0] + (targetGroup._size[0] / 2) - 300; // Center offset
                importOffsetY = targetGroup._pos[1] + (targetGroup._size[1] / 2) - 200;
            } else {
                // Import as new group - use mouse position
                importOffsetX = mousePos?.[0] ?? 300;
                importOffsetY = mousePos?.[1] ?? 300;
            }

            console.log(`📥 Importing ${nodes.length} nodes and ${links.length} links`);
            console.log(`📍 Import position: [${importOffsetX}, ${importOffsetY}]`);

            // Create nodes
            for (const n of nodes) {
                const node = LiteGraph.createNode(n.type);
                if (!node) {
                    console.warn(`⚠️ Could not create node of type: ${n.type}`);
                    continue;
                }

                node.configure(n);
                node.pos = [
                    (n.pos?.[0] || 0) + importOffsetX,
                    (n.pos?.[1] || 0) + importOffsetY
                ];

                app.graph.add(node);
                nodeMap[n.id] = node;
                console.log(`   Created node: ${n.type || n.title} at [${node.pos[0]}, ${node.pos[1]}]`);
            }

            // Reconnect links
            let connectionsMade = 0;
            for (const l of links) {
                const from = nodeMap[l[1]];
                const to = nodeMap[l[3]];
                if (!from || !to) {
                    console.warn(`⚠️ Could not connect link: from ${l[1]} to ${l[3]}`);
                    continue;
                }

                from.connect(l[2], to, l[4]);
                connectionsMade++;
            }
            console.log(`   Connected ${connectionsMade} links`);

            const importedNodes = Object.values(nodeMap);
            if (importedNodes.length > 0) {
                let targetGroupForNodes;
                
                if (targetGroup) {
                    // Use existing group
                    targetGroupForNodes = targetGroup;
                    console.log(`✅ Imported ${importedNodes.length} nodes into existing group "${targetGroup.title}"`);
                } else {
                    // Create new group
                    let GroupClass;
                    if (typeof LiteGraph.LGraphGroup === 'undefined') {
                        if (typeof LGraphGroup !== 'undefined') {
                            GroupClass = LGraphGroup;
                        } else {
                            console.error('❌ No group class found');
                            return;
                        }
                    } else {
                        GroupClass = LiteGraph.LGraphGroup;
                    }
                    
                    // Generate unique group name from filename
                    const baseName = file.name.replace(".json", "").replace(/_/g, " ") || "Imported Workflow";
                    const uniqueGroupName = getUniqueGroupName(baseName);
                    
                    const group = new GroupClass();
                    group.title = uniqueGroupName;
                    
                    // Calculate group bounds around imported nodes
                    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                    importedNodes.forEach(node => {
                        if (!node.pos) return;
                        const w = node.size?.[0] || 200;
                        const h = node.size?.[1] || 100;
                        minX = Math.min(minX, node.pos[0]);
                        minY = Math.min(minY, node.pos[1]);
                        maxX = Math.max(maxX, node.pos[0] + w);
                        maxY = Math.max(maxY, node.pos[1] + h);
                    });
                    
                    const padding = 40;
                    group.pos = [
                        Math.max(0, minX - padding),
                        Math.max(0, minY - padding)
                    ];
                    group.size = [
                        Math.max(400, (maxX - minX) + padding * 2),
                        Math.max(300, (maxY - minY) + padding * 2)
                    ];
                    
                    // Set color
                    if (typeof LGraphCanvas !== 'undefined' && LGraphCanvas.node_colors && LGraphCanvas.node_colors.pale_blue) {
                        group.color = LGraphCanvas.node_colors.pale_blue.color;
                    } else {
                        group.color = "#6495ED";
                    }
                    
                    app.graph.add(group);
                    targetGroupForNodes = group;
                    console.log(`✅ Created new group "${group.title}" with ${importedNodes.length} nodes`);
                }
                
                // Apply a grid layout to the imported nodes
                setTimeout(() => {
                    console.log('📐 Auto-arranging imported nodes in grid layout...');
                    arrangeGrid(importedNodes, targetGroupForNodes);
                    app.graph.setDirtyCanvas(true, true);
                }, 100);
            }

            app.graph.setDirtyCanvas(true, true);
            console.log('✅ Workflow import complete!');
            
        } catch (error) {
            console.error('❌ Error importing workflow:', error);
        }
    };

    input.click();
}

function createGroupForUngroupedNodes(canvas) {
    console.log('📦 Creating group for ungrouped nodes...');
    console.log('📊 Canvas object:', canvas ? 'Valid' : 'NULL');
    
    if (!canvas) {
        console.error('❌ No canvas provided');
        return;
    }
    
    const ungroupedNodes = getUngroupedNodes();
    console.log(`🔍 Found ${ungroupedNodes.length} ungrouped nodes`);
    
    if (!ungroupedNodes.length) {
        console.log('❌ No ungrouped nodes to group');
        return;
    }
    
    // Debug: List all found nodes
    ungroupedNodes.forEach((node, i) => {
        console.log(`   Node ${i}: ${node.type || node.title} at [${node.pos?.[0]}, ${node.pos?.[1]}]`);
    });
    
    // Calculate bounds for all ungrouped nodes
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let hasValidNodes = false;
    
    ungroupedNodes.forEach(node => {
        if (!node.pos) {
            console.log(`   Skipping node without position: ${node.type || node.title}`);
            return;
        }
        
        const w = node.size?.[0] || 200;
        const h = node.size?.[1] || 100;
        const x = node.pos[0];
        const y = node.pos[1];
        
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + w);
        maxY = Math.max(maxY, y + h);
        hasValidNodes = true;
    });
    
    if (!hasValidNodes) {
        console.error('❌ No nodes with valid positions found');
        return;
    }
    
    // Add padding around the nodes
    const padding = 40;
    const groupX = Math.max(0, minX - padding);
    const groupY = Math.max(0, minY - padding);
    const groupWidth = Math.max(400, (maxX - minX) + (padding * 2));
    const groupHeight = Math.max(300, (maxY - minY) + (padding * 2));
    
    console.log(`📏 Calculated group bounds:`, {
        pos: [groupX, groupY],
        size: [groupWidth, groupHeight],
        nodeBounds: { minX, minY, maxX, maxY }
    });
    
    try {
        // Create the group using the correct method
        console.log('Creating group using LiteGraph.LGraphGroup...');
        
        // Check if LiteGraph.LGraphGroup exists
        if (typeof LiteGraph === 'undefined') {
            console.error('❌ LiteGraph is not defined');
            return;
        }
        
        let GroupClass;
        if (typeof LiteGraph.LGraphGroup === 'undefined') {
            console.error('❌ LiteGraph.LGraphGroup is not defined');
            // Try alternative names
            if (typeof LGraphGroup !== 'undefined') {
                console.log('⚠️ Found LGraphGroup (without LiteGraph prefix)');
                GroupClass = LGraphGroup;
            } else {
                console.error('❌ No group class found');
                return;
            }
        } else {
            GroupClass = LiteGraph.LGraphGroup;
        }
        
        const group = new GroupClass();
        
        if (!group) {
            console.error('❌ Failed to create group node');
            return;
        }
        
        // Set group properties
        group.pos = [groupX, groupY];
        group.size = [groupWidth, groupHeight];
        group.title = "Grouped Nodes";
        
        // Set color - check if LGraphCanvas exists
        if (typeof LGraphCanvas !== 'undefined' && LGraphCanvas.node_colors && LGraphCanvas.node_colors.pale_blue) {
            group.color = LGraphCanvas.node_colors.pale_blue.color;
        } else {
            // Fallback color
            group.color = "#6495ED"; // Pale blue
        }
        
        console.log('Group created:', {
            type: group.constructor.name,
            pos: group.pos,
            size: group.size,
            title: group.title,
            color: group.color
        });

        arrangeGrid(ungroupedNodes, group);
        
        // Add group to graph
        if (app.graph && typeof app.graph.add === 'function') {
            console.log('Adding group to graph using app.graph.add()...');
            app.graph.add(group);
            console.log(`✅ Group added to graph`);
        } else {
            console.error('❌ Cannot add group to graph - app.graph.add not found');
            return;
        }
        
        // Verify the group was added
        console.log('Checking if group exists in graph...');
        if (app.graph._groups) {
            console.log(`Total groups in graph: ${app.graph._groups.length}`);
            const lastGroup = app.graph._groups[app.graph._groups.length - 1];
            if (lastGroup) {
                console.log('Last group in graph:', {
                    pos: lastGroup._pos || lastGroup.pos,
                    size: lastGroup._size || lastGroup.size,
                    title: lastGroup.title
                });
            }
        }
        
        console.log('🔄 Triggering canvas refresh...');
        app.graph.setDirtyCanvas(true, true);
        
        // Force a canvas redraw
        if (canvas && typeof canvas.draw === 'function') {
            setTimeout(() => {
                console.log('Forcing canvas redraw...');
                canvas.draw(true, true);
            }, 100);
        }
        
        console.log('✅ Group creation complete!');
        
    } catch (error) {
        console.error('❌ Error creating group:', error);
        console.error('Stack:', error.stack);
    }
}

function createGroupForSelectedNodes(canvas) {
    console.log('📦 Creating group for selected nodes...');
    
    if (!canvas) {
        console.error('❌ No canvas provided');
        return;
    }
    
    // Try multiple ways to get selected nodes
    let selectedNodes = [];
    
    // Method 1: Check canvas.selected_nodes
    if (canvas.selected_nodes && canvas.selected_nodes.length > 0) {
        selectedNodes = canvas.selected_nodes;
        console.log(`🔍 Found ${selectedNodes.length} selected nodes from canvas.selected_nodes`);
    }
    // Method 2: Check app.graph._nodes for nodes with selected flag
    else {
        selectedNodes = app.graph._nodes.filter(node => node.flags?.selected === true);
        console.log(`🔍 Found ${selectedNodes.length} selected nodes from app.graph._nodes flags`);
    }
    
    console.log(`🔍 Total selected nodes found: ${selectedNodes.length}`);
    
    if (selectedNodes.length === 0) {
        console.log('❌ No nodes selected');
        return;
    }
    
    // Debug: List all selected nodes
    selectedNodes.forEach((node, i) => {
        console.log(`   Selected Node ${i}: ${node.type || node.title || 'unknown'} at [${node.pos?.[0]}, ${node.pos?.[1]}]`);
    });
    
    // Calculate bounds for all selected nodes
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let hasValidNodes = false;
    
    selectedNodes.forEach(node => {
        if (!node.pos) {
            console.log(`   Skipping node without position: ${node.type || node.title || 'unknown'}`);
            return;
        }
        
        const w = node.size?.[0] || 200;
        const h = node.size?.[1] || 100;
        const x = node.pos[0];
        const y = node.pos[1];
        
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + w);
        maxY = Math.max(maxY, y + h);
        hasValidNodes = true;
    });
    
    if (!hasValidNodes) {
        console.error('❌ No selected nodes with valid positions found');
        return;
    }
    
    // Add padding around the nodes
    const padding = 40;
    const groupX = Math.max(0, minX - padding);
    const groupY = Math.max(0, minY - padding);
    const groupWidth = Math.max(400, (maxX - minX) + (padding * 2));
    const groupHeight = Math.max(300, (maxY - minY) + (padding * 2));
    
    console.log(`📏 Calculated group bounds:`, {
        pos: [groupX, groupY],
        size: [groupWidth, groupHeight],
        nodeBounds: { minX, minY, maxX, maxY }
    });
    
    try {
        // Create the group using the correct method
        console.log('Creating group using LiteGraph.LGraphGroup...');
        
        // Check if LiteGraph.LGraphGroup exists
        if (typeof LiteGraph === 'undefined') {
            console.error('❌ LiteGraph is not defined');
            return;
        }
        
        let GroupClass;
        if (typeof LiteGraph.LGraphGroup === 'undefined') {
            console.error('❌ LiteGraph.LGraphGroup is not defined');
            // Try alternative names
            if (typeof LGraphGroup !== 'undefined') {
                console.log('⚠️ Found LGraphGroup (without LiteGraph prefix)');
                GroupClass = LGraphGroup;
            } else {
                console.error('❌ No group class found');
                return;
            }
        } else {
            GroupClass = LiteGraph.LGraphGroup;
        }
        
        const group = new GroupClass();
        
        if (!group) {
            console.error('❌ Failed to create group node');
            return;
        }
        
        // Set group properties
        group.pos = [groupX, groupY];
        group.size = [groupWidth, groupHeight];
        group.title = "Selected Nodes Group";
        
        // Set color - check if LGraphCanvas exists
        if (typeof LGraphCanvas !== 'undefined' && LGraphCanvas.node_colors && LGraphCanvas.node_colors.pale_yellow) {
            group.color = LGraphCanvas.node_colors.pale_yellow.color;
        } else {
            // Fallback color
            group.color = "#F0E68C"; // Pale yellow for selected nodes
        }
        
        console.log('Group created:', {
            type: group.constructor.name,
            pos: group.pos,
            size: group.size,
            title: group.title,
            color: group.color
        });
        
        // Add group to graph
        if (app.graph && typeof app.graph.add === 'function') {
            console.log('Adding group to graph using app.graph.add()...');
            app.graph.add(group);
            console.log(`✅ Group added to graph`);
        } else {
            console.error('❌ Cannot add group to graph - app.graph.add not found');
            return;
        }
        
        // Clear selection after grouping
        if (canvas.clearSelection) {
            canvas.clearSelection();
        } else {
            // Manually clear selection flags
            selectedNodes.forEach(node => {
                if (node.flags) {
                    node.flags.selected = false;
                }
            });
            if (canvas.selected_nodes) {
                canvas.selected_nodes = [];
            }
        }
        
        console.log('🔄 Triggering canvas refresh...');
        app.graph.setDirtyCanvas(true, true);
        
        // Force a canvas redraw
        if (canvas && typeof canvas.draw === 'function') {
            setTimeout(() => {
                console.log('Forcing canvas redraw...');
                canvas.draw(true, true);
            }, 100);
        }
        
        console.log('✅ Selected nodes group creation complete!');
        
    } catch (error) {
        console.error('❌ Error creating group for selected nodes:', error);
        console.error('Stack:', error.stack);
    }
}

function arrangeGrid(nodes, group) {
    console.log('🔲 Grid layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    
    const nodeDimensions = nodes.map(node => ({
        width: node.size?.[0] || 200,
        height: node.size?.[1] || 100,
        node
    }));
    
    // Calculate optimal columns based on available space
    const maxNodeWidth = Math.max(...nodeDimensions.map(d => d.width));
    const maxNodeHeight = Math.max(...nodeDimensions.map(d => d.height));
    
    // Calculate optimal grid dimensions
    let bestCols = 1;
    let bestRows = nodes.length;
    let bestScore = Infinity;
    
    // Try different column counts to find the best aspect ratio match
    for (let cols = 1; cols <= Math.min(nodes.length, 10); cols++) {
        const rows = Math.ceil(nodes.length / cols);
        
        // Calculate cell dimensions (including spacing)
        const horizontalSpacing = 40;
        const verticalSpacing = 40;
        
        const totalWidth = cols * maxNodeWidth + (cols - 1) * horizontalSpacing;
        const totalHeight = rows * maxNodeHeight + (rows - 1) * verticalSpacing;
        
        // Calculate how well this fits the available space
        const widthFit = Math.max(0, bounds.availableWidth - totalWidth);
        const heightFit = Math.max(0, bounds.availableHeight - totalHeight);
        
        // Score: prefer layouts that fill space more evenly
        const aspectRatioDiff = Math.abs(
            (totalWidth / totalHeight) - (bounds.availableWidth / bounds.availableHeight)
        );
        const emptySpace = widthFit * heightFit;
        
        const score = aspectRatioDiff * 10 + emptySpace;
        
        if (score < bestScore) {
            bestScore = score;
            bestCols = cols;
            bestRows = rows;
        }
    }
    
    const cols = bestCols;
    const rows = bestRows;
    
    console.log(`📊 Grid dimensions: ${rows} rows x ${cols} columns`);
    console.log(`📏 Available space: ${bounds.availableWidth} x ${bounds.availableHeight}`);
    
    // Calculate actual cell dimensions based on individual node sizes
    const colWidths = new Array(cols).fill(0);
    const rowHeights = new Array(rows).fill(0);
    
    for (let i = 0; i < nodes.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const dim = nodeDimensions[i];
        
        colWidths[col] = Math.max(colWidths[col], dim.width);
        rowHeights[row] = Math.max(rowHeights[row], dim.height);
    }
    
    // Calculate spacing - distribute remaining space evenly
    const totalContentWidth = colWidths.reduce((sum, width) => sum + width, 0);
    const totalContentHeight = rowHeights.reduce((sum, height) => sum + height, 0);
    
    let horizontalSpacing = 40; // Default minimum spacing
    let verticalSpacing = 40;
    
    // If we have extra space, distribute it
    if (cols > 1) {
        const extraWidth = bounds.availableWidth - totalContentWidth;
        horizontalSpacing = Math.max(40, extraWidth / (cols - 1));
    }
    
    if (rows > 1) {
        const extraHeight = bounds.availableHeight - totalContentHeight;
        verticalSpacing = Math.max(40, extraHeight / (rows - 1));
    }
    
    const totalGridWidth = totalContentWidth + (cols - 1) * horizontalSpacing;
    const totalGridHeight = totalContentHeight + (rows - 1) * verticalSpacing;
    
    // Center the grid within the available space
    let gridStartX = bounds.startX + Math.max(0, (bounds.availableWidth - totalGridWidth) / 2);
    let gridStartY = bounds.startY + Math.max(0, (bounds.availableHeight - totalGridHeight) / 2);
    
    // Ensure we stay within bounds
    gridStartX = Math.max(bounds.startX, gridStartX);
    gridStartY = Math.max(bounds.startY, gridStartY);
    
    console.log(`📏 Grid positioning:`, {
        totalGridWidth: totalGridWidth.toFixed(1),
        totalGridHeight: totalGridHeight.toFixed(1),
        gridStartX: gridStartX.toFixed(1),
        gridStartY: gridStartY.toFixed(1),
        horizontalSpacing: horizontalSpacing.toFixed(1),
        verticalSpacing: verticalSpacing.toFixed(1)
    });
    
    // Position each node
    for (let i = 0; i < nodes.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const node = nodes[i];
        const dim = nodeDimensions[i];
        
        // Calculate cell position
        let cellX = gridStartX;
        for (let c = 0; c < col; c++) {
            cellX += colWidths[c] + horizontalSpacing;
        }
        
        let cellY = gridStartY;
        for (let r = 0; r < row; r++) {
            cellY += rowHeights[r] + verticalSpacing;
        }
        
        // Center node within its cell
        const oldPos = [...node.pos];
        node.pos[0] = cellX + (colWidths[col] - dim.width) / 2;
        node.pos[1] = cellY + (rowHeights[row] - dim.height) / 2;
        
        console.log(`   Node ${i} (row ${row}, col ${col}):`, {
            type: node.type || node.title,
            oldPos: oldPos.map(p => p.toFixed(1)),
            newPos: node.pos.map(p => p.toFixed(1)),
            cell: { x: cellX.toFixed(1), y: cellY.toFixed(1) }
        });
        
        // Apply boundary check
        applyBoundaryCheck(node, bounds, dim.width, dim.height);
    }
    
    console.log('✅ Grid layout complete');
}

function arrangeTightGrid(nodes, group) {
    console.log('🔷 Tight grid layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const maxSpacing = 20; // Maximum 20px spacing for tight packing
    
    const nodeDimensions = nodes.map(node => ({
        width: node.size?.[0] || 200,
        height: node.size?.[1] || 100,
        node
    }));
    
    const maxNodeWidth = Math.max(...nodeDimensions.map(d => d.width));
    const maxNodeHeight = Math.max(...nodeDimensions.map(d => d.height));
    
    // Calculate optimal grid dimensions
    // Try to make grid as square as possible
    const nodeCount = nodes.length;
    let bestCols = Math.ceil(Math.sqrt(nodeCount));
    let bestRows = Math.ceil(nodeCount / bestCols);
    
    // Try to find better aspect ratio if possible
    for (let cols = 1; cols <= Math.min(nodeCount, 15); cols++) {
        const rows = Math.ceil(nodeCount / cols);
        
        // Calculate grid dimensions with max 20px spacing
        const gridWidth = cols * maxNodeWidth + (cols - 1) * maxSpacing;
        const gridHeight = rows * maxNodeHeight + (rows - 1) * maxSpacing;
        
        // Check if it fits within bounds
        if (gridWidth <= bounds.availableWidth && gridHeight <= bounds.availableHeight) {
            // Calculate aspect ratio difference (closer to 1 is more square)
            const aspectRatioDiff = Math.abs(cols - rows);
            const currentBestDiff = Math.abs(bestCols - bestRows);
            
            // Prefer more square layouts that fit
            if (aspectRatioDiff < currentBestDiff) {
                bestCols = cols;
                bestRows = rows;
            }
        }
    }
    
    // If we couldn't find a layout that fits with max spacing, use the sqrt-based one
    // and adjust spacing to fit
    const cols = bestCols;
    const rows = bestRows;
    
    console.log(`📊 Tight grid dimensions: ${rows} rows x ${cols} columns for ${nodeCount} nodes`);
    
    // Calculate column widths and row heights based on actual node sizes
    const colWidths = new Array(cols).fill(0);
    const rowHeights = new Array(rows).fill(0);
    
    for (let i = 0; i < nodes.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const dim = nodeDimensions[i];
        
        colWidths[col] = Math.max(colWidths[col], dim.width);
        rowHeights[row] = Math.max(rowHeights[row], dim.height);
    }
    
    // Calculate total content size
    const totalContentWidth = colWidths.reduce((sum, width) => sum + width, 0);
    const totalContentHeight = rowHeights.reduce((sum, height) => sum + height, 0);
    
    // Calculate maximum allowed spacing based on available space
    let horizontalSpacing = maxSpacing;
    let verticalSpacing = maxSpacing;
    
    if (cols > 1) {
        const maxPossibleHSpace = Math.max(0, bounds.availableWidth - totalContentWidth) / (cols - 1);
        horizontalSpacing = Math.min(maxSpacing, maxPossibleHSpace);
    }
    
    if (rows > 1) {
        const maxPossibleVSpace = Math.max(0, bounds.availableHeight - totalContentHeight) / (rows - 1);
        verticalSpacing = Math.min(maxSpacing, maxPossibleVSpace);
    }
    
    const totalGridWidth = totalContentWidth + (cols - 1) * horizontalSpacing;
    const totalGridHeight = totalContentHeight + (rows - 1) * verticalSpacing;
    
    // Center the grid within available space
    let gridStartX = bounds.startX + Math.max(0, (bounds.availableWidth - totalGridWidth) / 2);
    let gridStartY = bounds.startY + Math.max(0, (bounds.availableHeight - totalGridHeight) / 2);
    
    // Ensure we stay within bounds
    gridStartX = Math.max(bounds.startX, gridStartX);
    gridStartY = Math.max(bounds.startY, gridStartY);
    
    console.log(`📏 Tight grid positioning:`, {
        totalGridWidth: totalGridWidth.toFixed(1),
        totalGridHeight: totalGridHeight.toFixed(1),
        gridStartX: gridStartX.toFixed(1),
        gridStartY: gridStartY.toFixed(1),
        horizontalSpacing: horizontalSpacing.toFixed(1),
        verticalSpacing: verticalSpacing.toFixed(1),
        maxSpacing
    });
    
    // Position each node
    for (let i = 0; i < nodes.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const node = nodes[i];
        const dim = nodeDimensions[i];
        
        // Calculate cell position
        let cellX = gridStartX;
        for (let c = 0; c < col; c++) {
            cellX += colWidths[c] + horizontalSpacing;
        }
        
        let cellY = gridStartY;
        for (let r = 0; r < row; r++) {
            cellY += rowHeights[r] + verticalSpacing;
        }
        
        // Center node within its cell
        const oldPos = [...node.pos];
        node.pos[0] = cellX + (colWidths[col] - dim.width) / 2;
        node.pos[1] = cellY + (rowHeights[row] - dim.height) / 2;
        
        console.log(`   Node ${i} (row ${row}, col ${col}):`, {
            type: node.type || node.title,
            oldPos: oldPos.map(p => p.toFixed(1)),
            newPos: node.pos.map(p => p.toFixed(1)),
            size: { width: dim.width, height: dim.height },
            spacing: { h: horizontalSpacing.toFixed(1), v: verticalSpacing.toFixed(1) }
        });
        
        applyBoundaryCheck(node, bounds, dim.width, dim.height);
    }
    
    console.log('✅ Tight grid layout complete with max 20px spacing');
}

function arrangeHorizontal(nodes, group) {
    console.log('➡️ Horizontal layout starting...');
    
    const spacing = 40;
    const bounds = calculateLayoutBounds(group);
    
    const totalWidth = nodes.reduce((sum, node) => sum + (node.size?.[0] || 200), 0) + 
                      (nodes.length - 1) * spacing;
    
    let currentX = bounds.startX + Math.max(0, (bounds.availableWidth - totalWidth) / 2);
    const centerY = bounds.startY + bounds.availableHeight / 2;
    
    console.log(`📏 Horizontal layout:`, {
        totalWidth,
        currentX,
        centerY
    });
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        const oldPos = [...node.pos];
        node.pos[0] = currentX;
        node.pos[1] = centerY - height / 2;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            size: { width, height }
        });
        
        applyBoundaryCheck(node, bounds, width, height);
        
        currentX += width + spacing;
    });
    
    console.log('✅ Horizontal layout complete');
}

function arrangeVertical(nodes, group) {
    console.log('⬇️ Vertical layout starting...');
    
    const spacing = 40;
    const bounds = calculateLayoutBounds(group);
    
    const totalHeight = nodes.reduce((sum, node) => sum + (node.size?.[1] || 100), 0) + 
                       (nodes.length - 1) * spacing;
    
    const centerX = bounds.startX + bounds.availableWidth / 2;
    let currentY = bounds.startY + Math.max(0, (bounds.availableHeight - totalHeight) / 2);
    
    console.log(`📏 Vertical layout:`, {
        totalHeight,
        centerX,
        currentY
    });
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        const oldPos = [...node.pos];
        node.pos[0] = centerX - width / 2;
        node.pos[1] = currentY;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            size: { width, height }
        });
        
        applyBoundaryCheck(node, bounds, width, height);
        
        currentY += height + spacing;
    });
    
    console.log('✅ Vertical layout complete');
}

function arrangeCircular(nodes, group) {
    console.log('⭕ Circular layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate maximum radius that fits within group with 10px margin
    const margin = 10; // 10px from all edges
    
    // Calculate available space for circle (accounting for title bar and margins)
    const availableWidth = group._size[0] - (2 * margin);
    const availableHeight = group._size[1] - (2 * margin) - 30; // -30 for title bar
    
    // Get the maximum node dimension to ensure nodes don't overlap boundaries
    const maxNodeWidth = Math.max(...nodes.map(n => n.size?.[0] || 200));
    const maxNodeHeight = Math.max(...nodes.map(n => n.size?.[1] || 100));
    const maxNodeRadius = Math.sqrt(maxNodeWidth * maxNodeWidth + maxNodeHeight * maxNodeHeight) / 2;
    
    // Calculate safe radius (half of the smallest dimension minus node radius and margin)
    const maxRadius = Math.min(availableWidth, availableHeight) / 2 - maxNodeRadius;
    
    // Ensure minimum radius
    const r = Math.max(50, maxRadius * 0.9); // Use 90% of maximum to be safe
    
    console.log(`📏 Circular layout:`, {
        centerX,
        centerY,
        radius: r,
        availableWidth,
        availableHeight,
        maxNodeRadius,
        margin,
        effectiveRadius: `Max possible: ${maxRadius}, Using: ${r}`
    });
    
    nodes.forEach((node, i) => {
        const angle = (i / nodes.length) * Math.PI * 2;
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        // Calculate position on circle
        const circleX = centerX + Math.cos(angle) * r;
        const circleY = centerY + Math.sin(angle) * r;
        
        // Position node so its center is on the circle
        const oldPos = [...node.pos];
        node.pos[0] = circleX - width / 2;
        node.pos[1] = circleY - height / 2;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            size: { width, height },
            angle: (angle * 180 / Math.PI).toFixed(1) + '°',
            circlePos: [circleX, circleY]
        });
        
        // Apply boundary check (now with 10px margin)
        applyBoundaryCheck(node, bounds, width, height);
        
        // Additional check specifically for circular layout
        const nodeRight = node.pos[0] + width;
        const nodeBottom = node.pos[1] + height;
        const groupLeft = group._pos[0] + margin;
        const groupTop = group._pos[1] + margin + 30; // Title bar
        const groupRight = group._pos[0] + group._size[0] - margin;
        const groupBottom = group._pos[1] + group._size[1] - margin;
        
        let needsAdjustment = false;
        
        if (node.pos[0] < groupLeft) {
            console.log(`   ⚠️ Node too far left, adjusting`);
            node.pos[0] = groupLeft;
            needsAdjustment = true;
        }
        if (node.pos[1] < groupTop) {
            console.log(`   ⚠️ Node too far up, adjusting`);
            node.pos[1] = groupTop;
            needsAdjustment = true;
        }
        if (nodeRight > groupRight) {
            console.log(`   ⚠️ Node too far right, adjusting`);
            node.pos[0] = groupRight - width;
            needsAdjustment = true;
        }
        if (nodeBottom > groupBottom) {
            console.log(`   ⚠️ Node too far down, adjusting`);
            node.pos[1] = groupBottom - height;
            needsAdjustment = true;
        }
        
        if (needsAdjustment) {
            console.log(`   ↪️ Final adjusted position: [${node.pos[0]}, ${node.pos[1]}]`);
        }
    });
    
    console.log('✅ Circular layout complete with 10px margin');
}

function arrangeHierarchyHorizontal(nodes, group) {
    console.log('🌳 Horizontal hierarchy layout starting...');
    
    const spacing = 40;
    const bounds = calculateLayoutBounds(group);
    
    const nodeMap = new Map();
    nodes.forEach((node, index) => {
        nodeMap.set(node.id, {
            index,
            node,
            inputs: [],
            outputs: [],
            level: -1,
            visited: false
        });
    });
    
    nodes.forEach((node) => {
        const nodeInfo = nodeMap.get(node.id);
        
        if (node.inputs) {
            node.inputs.forEach((input) => {
                if (input && input.link) {
                    const link = app.graph.links[input.link];
                    if (link && link.origin_id !== null) {
                        const sourceNode = nodeMap.get(link.origin_id);
                        if (sourceNode) {
                            sourceNode.outputs.push(node.id);
                            nodeInfo.inputs.push(link.origin_id);
                        }
                    }
                }
            });
        }
        
        if (node.outputs) {
            node.outputs.forEach((output) => {
                if (output && output.links) {
                    output.links.forEach((linkId) => {
                        const link = app.graph.links[linkId];
                        if (link && link.target_id !== null) {
                            const targetNode = nodeMap.get(link.target_id);
                            if (targetNode) {
                                nodeInfo.outputs.push(link.target_id);
                                targetNode.inputs.push(node.id);
                            }
                        }
                    });
                }
            });
        }
    });
    
    const rootNodes = [];
    nodeMap.forEach((nodeInfo, nodeId) => {
        if (nodeInfo.inputs.length === 0) {
            rootNodes.push(nodeId);
        }
    });
    
    console.log(`📊 Found ${rootNodes.length} root nodes`);
    
    let currentLevel = 0;
    const queue = [...rootNodes];
    
    rootNodes.forEach(nodeId => {
        const nodeInfo = nodeMap.get(nodeId);
        nodeInfo.level = 0;
        nodeInfo.visited = true;
    });
    
    while (queue.length > 0) {
        const currentId = queue.shift();
        const currentInfo = nodeMap.get(currentId);
        
        currentInfo.outputs.forEach(outputId => {
            const outputInfo = nodeMap.get(outputId);
            if (!outputInfo.visited) {
                outputInfo.level = currentInfo.level + 1;
                outputInfo.visited = true;
                queue.push(outputId);
            }
        });
    }
    
    let maxLevel = 0;
    nodeMap.forEach((nodeInfo) => {
        if (nodeInfo.level === -1) {
            nodeInfo.level = 0;
        }
        maxLevel = Math.max(maxLevel, nodeInfo.level);
    });
    
    const levels = Array(maxLevel + 1).fill().map(() => []);
    nodeMap.forEach((nodeInfo) => {
        levels[nodeInfo.level].push({
            id: nodeInfo.node.id,
            node: nodeInfo.node,
            width: nodeInfo.node.size?.[0] || 200,
            height: nodeInfo.node.size?.[1] || 100
        });
    });
    
    console.log(`📊 Hierarchy levels: ${levels.length}`, levels.map((levelNodes, idx) => 
        `Level ${idx}: ${levelNodes.length} nodes`));
    
    const levelWidths = levels.map(levelNodes => 
        Math.max(...levelNodes.map(n => n.width))
    );
    
    const maxRows = Math.max(...levels.map(levelNodes => levelNodes.length));
    
    const totalWidth = levelWidths.reduce((sum, width) => sum + width + spacing, -spacing);
    const totalHeight = maxRows * 100 + (maxRows - 1) * spacing;
    
    let startX = bounds.startX + Math.max(0, (bounds.availableWidth - totalWidth) / 2);
    let startY = bounds.startY + Math.max(0, (bounds.availableHeight - totalHeight) / 2);
    
    console.log(`📏 Hierarchy layout:`, {
        totalWidth,
        totalHeight,
        startX,
        startY,
        levelWidths,
        maxRows
    });
    
    let currentX = startX;
    
    for (let level = 0; level < levels.length; level++) {
        const levelNodes = levels[level];
        const levelWidth = levelWidths[level];
        
        const levelHeight = levelNodes.reduce((sum, n) => sum + n.height, 0) + 
                           (levelNodes.length - 1) * spacing;
        
        let currentY = startY + Math.max(0, (totalHeight - levelHeight) / 2);
        
        console.log(`   Level ${level}: ${levelNodes.length} nodes`, {
            currentX,
            levelWidth,
            levelHeight
        });
        
        levelNodes.sort((a, b) => {
            const aInfo = nodeMap.get(a.id);
            const bInfo = nodeMap.get(b.id);
            return bInfo.inputs.length - aInfo.inputs.length;
        });
        
        levelNodes.forEach((nodeData, idx) => {
            const node = nodeData.node;
            const width = nodeData.width;
            const height = nodeData.height;
            
            const xInLevel = currentX + (levelWidth - width) / 2;
            
            const oldPos = [...node.pos];
            node.pos[0] = xInLevel;
            node.pos[1] = currentY;
            
            console.log(`     Node ${idx}: ${node.type || node.title}`, {
                oldPos,
                newPos: node.pos,
                size: { width, height }
            });
            
            applyBoundaryCheck(node, bounds, width, height);
            
            currentY += height + spacing;
        });
        
        currentX += levelWidth + spacing;
    }
    
    console.log('✅ Horizontal hierarchy layout complete');
}

function arrangeHierarchyVertical(nodes, group) {
    console.log('🌳 Vertical hierarchy layout starting...');
    
    const spacing = 40;
    const bounds = calculateLayoutBounds(group);
    
    const nodeMap = new Map();
    nodes.forEach((node, index) => {
        nodeMap.set(node.id, {
            index,
            node,
            inputs: [],
            outputs: [],
            level: -1,
            visited: false
        });
    });
    
    nodes.forEach((node) => {
        const nodeInfo = nodeMap.get(node.id);
        
        if (node.inputs) {
            node.inputs.forEach((input) => {
                if (input && input.link) {
                    const link = app.graph.links[input.link];
                    if (link && link.origin_id !== null) {
                        const sourceNode = nodeMap.get(link.origin_id);
                        if (sourceNode) {
                            sourceNode.outputs.push(node.id);
                            nodeInfo.inputs.push(link.origin_id);
                        }
                    }
                }
            });
        }
        
        if (node.outputs) {
            node.outputs.forEach((output) => {
                if (output && output.links) {
                    output.links.forEach((linkId) => {
                        const link = app.graph.links[linkId];
                        if (link && link.target_id !== null) {
                            const targetNode = nodeMap.get(link.target_id);
                            if (targetNode) {
                                nodeInfo.outputs.push(link.target_id);
                                targetNode.inputs.push(node.id);
                            }
                        }
                    });
                }
            });
        }
    });
    
    const rootNodes = [];
    nodeMap.forEach((nodeInfo, nodeId) => {
        if (nodeInfo.inputs.length === 0) {
            rootNodes.push(nodeId);
        }
    });
    
    let currentLevel = 0;
    const queue = [...rootNodes];
    
    rootNodes.forEach(nodeId => {
        const nodeInfo = nodeMap.get(nodeId);
        nodeInfo.level = 0;
        nodeInfo.visited = true;
    });
    
    while (queue.length > 0) {
        const currentId = queue.shift();
        const currentInfo = nodeMap.get(currentId);
        
        currentInfo.outputs.forEach(outputId => {
            const outputInfo = nodeMap.get(outputId);
            if (!outputInfo.visited) {
                outputInfo.level = currentInfo.level + 1;
                outputInfo.visited = true;
                queue.push(outputId);
            }
        });
    }
    
    let maxLevel = 0;
    nodeMap.forEach((nodeInfo) => {
        if (nodeInfo.level === -1) {
            nodeInfo.level = 0;
        }
        maxLevel = Math.max(maxLevel, nodeInfo.level);
    });
    
    const levels = Array(maxLevel + 1).fill().map(() => []);
    nodeMap.forEach((nodeInfo) => {
        levels[nodeInfo.level].push({
            id: nodeInfo.node.id,
            node: nodeInfo.node,
            width: nodeInfo.node.size?.[0] || 200,
            height: nodeInfo.node.size?.[1] || 100
        });
    });
    
    console.log(`📊 Hierarchy levels: ${levels.length}`, levels.map((levelNodes, idx) => 
        `Level ${idx}: ${levelNodes.length} nodes`));
    
    const levelHeights = levels.map(levelNodes => 
        Math.max(...levelNodes.map(n => n.height))
    );
    
    const maxCols = Math.max(...levels.map(levelNodes => levelNodes.length));
    
    const totalHeight = levelHeights.reduce((sum, height) => sum + height + spacing, -spacing);
    const totalWidth = maxCols * 200 + (maxCols - 1) * spacing;
    
    let startX = bounds.startX + Math.max(0, (bounds.availableWidth - totalWidth) / 2);
    let startY = bounds.startY + Math.max(0, (bounds.availableHeight - totalHeight) / 2);
    
    console.log(`📏 Vertical hierarchy layout:`, {
        totalWidth,
        totalHeight,
        startX,
        startY,
        levelHeights,
        maxCols
    });
    
    let currentY = startY;
    
    for (let level = 0; level < levels.length; level++) {
        const levelNodes = levels[level];
        const levelHeight = levelHeights[level];
        
        const levelWidth = levelNodes.reduce((sum, n) => sum + n.width, 0) + 
                          (levelNodes.length - 1) * spacing;
        
        let currentX = startX + Math.max(0, (totalWidth - levelWidth) / 2);
        
        console.log(`   Level ${level}: ${levelNodes.length} nodes`, {
            currentY,
            levelHeight,
            levelWidth
        });
        
        levelNodes.sort((a, b) => {
            const aInfo = nodeMap.get(a.id);
            const bInfo = nodeMap.get(b.id);
            return bInfo.inputs.length - aInfo.inputs.length;
        });
        
        levelNodes.forEach((nodeData, idx) => {
            const node = nodeData.node;
            const width = nodeData.width;
            const height = nodeData.height;
            
            const yInLevel = currentY + (levelHeight - height) / 2;
            
            const oldPos = [...node.pos];
            node.pos[0] = currentX;
            node.pos[1] = yInLevel;
            
            console.log(`     Node ${idx}: ${node.type || node.title}`, {
                oldPos,
                newPos: node.pos,
                size: { width, height }
            });
            
            applyBoundaryCheck(node, bounds, width, height);
            
            currentX += width + spacing;
        });
        
        currentY += levelHeight + spacing;
    }
    
    console.log('✅ Vertical hierarchy layout complete');
}

function arrangeTriangle(nodes, group) {
    console.log('🔺 Triangle perimeter layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate triangle dimensions based on group size
    const margin = 80;
    const triangleHeight = Math.min(group._size[0], group._size[1]) * 0.7 - margin * 2;
    const triangleBase = triangleHeight * 1.732; // Equilateral triangle: height = √3/2 * base
    
    // Calculate triangle vertices (equilateral triangle pointing up)
    const topVertex = [centerX, centerY - triangleHeight / 2];
    const leftVertex = [centerX - triangleBase / 2, centerY + triangleHeight / 2];
    const rightVertex = [centerX + triangleBase / 2, centerY + triangleHeight / 2];
    
    // Calculate perimeter of triangle
    const sideLength = triangleBase;
    const trianglePerimeter = sideLength * 3;
    const spacing = trianglePerimeter / nodes.length;
    
    console.log(`📏 Triangle perimeter layout:`, {
        centerX, centerY,
        triangleHeight, triangleBase,
        vertices: { top: topVertex, left: leftVertex, right: rightVertex },
        perimeter: trianglePerimeter,
        spacing,
        nodes: nodes.length
    });
    
    let distanceTraveled = 0;
    const sideLengthPerNode = sideLength / Math.ceil(nodes.length / 3);
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        let x, y;
        
        // Determine which side of the triangle we're on
        const side = Math.floor(distanceTraveled / sideLength);
        const sidePosition = distanceTraveled % sideLength;
        const normalizedPos = sidePosition / sideLength;
        
        switch(side % 3) {
            case 0: // Left side (top to left)
                x = topVertex[0] + (leftVertex[0] - topVertex[0]) * normalizedPos;
                y = topVertex[1] + (leftVertex[1] - topVertex[1]) * normalizedPos;
                break;
            case 1: // Bottom side (left to right)
                x = leftVertex[0] + (rightVertex[0] - leftVertex[0]) * normalizedPos;
                y = leftVertex[1] + (rightVertex[1] - leftVertex[1]) * normalizedPos;
                break;
            case 2: // Right side (right to top)
                x = rightVertex[0] + (topVertex[0] - rightVertex[0]) * normalizedPos;
                y = rightVertex[1] + (topVertex[1] - rightVertex[1]) * normalizedPos;
                break;
        }
        
        // Center the node on the triangle perimeter
        x -= width / 2;
        y -= height / 2;
        
        const oldPos = [...node.pos];
        node.pos[0] = x;
        node.pos[1] = y;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            size: { width, height },
            side: ['left', 'bottom', 'right'][side % 3],
            position: normalizedPos.toFixed(2)
        });
        
        // Apply boundary check
        applyBoundaryCheck(node, bounds, width, height);
        
        distanceTraveled += spacing;
    });
    
    console.log('✅ Triangle perimeter layout complete');
}

function arrangeSquare(nodes, group) {
    console.log('⬛ Square perimeter layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate square/rectangle dimensions based on group size
    const margin = 80;
    const squareWidth = group._size[0] - margin * 2;
    const squareHeight = group._size[1] - margin * 2 - 30; // Account for title bar
    
    // Place nodes along the perimeter (nothing in center)
    const perimeter = 2 * (squareWidth + squareHeight);
    const spacing = perimeter / nodes.length;
    
    console.log(`📏 Square perimeter layout:`, {
        centerX, centerY,
        squareWidth, squareHeight,
        perimeter, spacing
    });
    
    let distanceTraveled = 0;
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        let x, y;
        
        // Determine which side of the square we're on
        if (distanceTraveled < squareWidth) {
            // Top side (left to right)
            x = centerX - squareWidth / 2 + distanceTraveled;
            y = centerY - squareHeight / 2 - height / 2;
        } else if (distanceTraveled < squareWidth + squareHeight) {
            // Right side (top to bottom)
            x = centerX + squareWidth / 2 - width / 2;
            y = centerY - squareHeight / 2 + (distanceTraveled - squareWidth);
        } else if (distanceTraveled < 2 * squareWidth + squareHeight) {
            // Bottom side (right to left)
            x = centerX + squareWidth / 2 - (distanceTraveled - squareWidth - squareHeight);
            y = centerY + squareHeight / 2 - height / 2;
        } else {
            // Left side (bottom to top)
            x = centerX - squareWidth / 2 - width / 2;
            y = centerY + squareHeight / 2 - (distanceTraveled - 2 * squareWidth - squareHeight);
        }
        
        const oldPos = [...node.pos];
        node.pos[0] = x;
        node.pos[1] = y;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            side: distanceTraveled < squareWidth ? "top" : 
                  distanceTraveled < squareWidth + squareHeight ? "right" :
                  distanceTraveled < 2 * squareWidth + squareHeight ? "bottom" : "left"
        });
        
        applyBoundaryCheck(node, bounds, width, height);
        distanceTraveled += spacing;
    });
    
    console.log('✅ Square perimeter layout complete');
}

function arrangeDiamond(nodes, group) {
    console.log('🔷 Diamond layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate diamond dimensions based on group size
    const margin = 100;
    const diamondWidth = Math.min(group._size[0], group._size[1]) - margin * 2;
    const diamondHeight = diamondWidth;
    
    // Calculate points along diamond perimeter
    const diamondPerimeter = 4 * Math.sqrt(2) * (diamondWidth / 2);
    const spacing = diamondPerimeter / nodes.length;
    
    console.log(`📏 Diamond layout:`, {
        centerX, centerY,
        diamondWidth, diamondHeight,
        perimeter: diamondPerimeter,
        spacing
    });
    
    let distanceTraveled = 0;
    const halfWidth = diamondWidth / 2;
    const quarterPerimeter = diamondPerimeter / 4;
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        let x, y;
        const segment = Math.floor(distanceTraveled / quarterPerimeter);
        const segmentPos = distanceTraveled % quarterPerimeter;
        const normalizedPos = segmentPos / quarterPerimeter;
        
        switch(segment) {
            case 0: // Top right quadrant
                x = centerX + normalizedPos * halfWidth;
                y = centerY - (1 - normalizedPos) * halfWidth;
                break;
            case 1: // Bottom right quadrant
                x = centerX + (1 - normalizedPos) * halfWidth;
                y = centerY + normalizedPos * halfWidth;
                break;
            case 2: // Bottom left quadrant
                x = centerX - normalizedPos * halfWidth;
                y = centerY + (1 - normalizedPos) * halfWidth;
                break;
            case 3: // Top left quadrant
                x = centerX - (1 - normalizedPos) * halfWidth;
                y = centerY - normalizedPos * halfWidth;
                break;
        }
        
        // Center the node on the diamond point
        x -= width / 2;
        y -= height / 2;
        
        const oldPos = [...node.pos];
        node.pos[0] = x;
        node.pos[1] = y;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            quadrant: segment
        });
        
        applyBoundaryCheck(node, bounds, width, height);
        distanceTraveled += spacing;
    });
    
    console.log('✅ Diamond layout complete');
}

function arrangeFillGroup(nodes, group) {
    console.log('📦 Fill group layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    
    // Sort nodes by area (largest first for better packing)
    const sortedNodes = nodes.slice().sort((a, b) => {
        const aArea = (a.size?.[0] || 200) * (a.size?.[1] || 100);
        const bArea = (b.size?.[0] || 200) * (b.size?.[1] || 100);
        return bArea - aArea; // Largest first
    });
    
    console.log(`📏 Fill group layout: ${sortedNodes.length} nodes in ${bounds.availableWidth}x${bounds.availableHeight} area`);
    
    const placedNodes = [];
    let currentX = bounds.startX;
    let currentY = bounds.startY;
    let rowHeight = 0;
    let rowStartY = currentY;
    
    // Add some padding between nodes
    const horizontalPadding = 20;
    const verticalPadding = 20;
    
    sortedNodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        // Check if node fits in current row
        if (currentX + width > bounds.groupRight) {
            // Move to next row
            currentX = bounds.startX;
            currentY = rowStartY + rowHeight + verticalPadding;
            rowStartY = currentY;
            rowHeight = 0;
            
            console.log(`   ↪️ Moving to next row at Y=${currentY}`);
        }
        
        // Check if we need to move to a new column (if node is taller than available space)
        if (currentY + height > bounds.groupBottom) {
            console.log(`   ⚠️ Node ${i} doesn't fit in remaining vertical space, placing at start`);
            currentX = bounds.startX;
            currentY = bounds.startY;
            rowStartY = currentY;
            rowHeight = 0;
        }
        
        // Place the node
        const oldPos = [...node.pos];
        node.pos[0] = currentX;
        node.pos[1] = currentY;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            size: { width, height },
            row: `X=${currentX}, Y=${currentY}`
        });
        
        placedNodes.push({
            node,
            x: currentX,
            y: currentY,
            width,
            height
        });
        
        // Update current position for next node
        currentX += width + horizontalPadding;
        rowHeight = Math.max(rowHeight, height);
        
        // Apply boundary check
        applyBoundaryCheck(node, bounds, width, height);
    });
    
    // Optional: Try to compact the layout by moving nodes up
    if (placedNodes.length > 1) {
        console.log('   Compacting layout...');
        
        // Sort by Y position, then X
        placedNodes.sort((a, b) => {
            if (a.y !== b.y) return a.y - b.y;
            return a.x - b.x;
        });
        
        // Group by rows (similar Y positions)
        const rows = [];
        let currentRow = [];
        let currentRowY = placedNodes[0].y;
        const rowTolerance = 10; // Nodes within 10px Y difference are considered same row
        
        placedNodes.forEach(placedNode => {
            if (Math.abs(placedNode.y - currentRowY) > rowTolerance) {
                if (currentRow.length > 0) {
                    rows.push([...currentRow]);
                }
                currentRow = [placedNode];
                currentRowY = placedNode.y;
            } else {
                currentRow.push(placedNode);
            }
        });
        
        if (currentRow.length > 0) {
            rows.push(currentRow);
        }
        
        // Compact each row
        rows.forEach((row, rowIndex) => {
            row.sort((a, b) => a.x - b.x);
            
            // Start from the beginning of the row
            let compactX = bounds.startX;
            
            row.forEach(placedNode => {
                const node = placedNode.node;
                const width = placedNode.width;
                
                // Only move if we can save space
                if (node.pos[0] > compactX + 5) { // Only move if more than 5px difference
                    const oldX = node.pos[0];
                    node.pos[0] = compactX;
                    console.log(`     Compacted ${node.type || node.title}: X ${oldX} → ${compactX}`);
                }
                
                compactX += width + horizontalPadding;
            });
        });
    }
    
    console.log('✅ Fill group layout complete');
}

function arrangeHeart(nodes, group) {
    console.log('❤️ Heart shape layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate heart dimensions
    const heartSize = Math.min(group._size[0], group._size[1]) * 0.4;
    const spacing = (2 * Math.PI * heartSize) / nodes.length;
    
    console.log(`📏 Heart layout:`, {
        centerX, centerY,
        heartSize,
        spacing
    });
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        // Parametric heart equation
        const t = (i / nodes.length) * 2 * Math.PI;
        const x = 16 * Math.pow(Math.sin(t), 3);
        const y = 13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t);
        
        // Scale and position
        const scaledX = centerX + x * heartSize / 20;
        const scaledY = centerY - y * heartSize / 20; // Negative because canvas Y increases downward
        
        // Center node
        const oldPos = [...node.pos];
        node.pos[0] = scaledX - width / 2;
        node.pos[1] = scaledY - height / 2;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            t: (t * 180 / Math.PI).toFixed(1) + '°'
        });
        
        applyBoundaryCheck(node, bounds, width, height);
    });
    
    console.log('✅ Heart shape layout complete');
}

function arrangeStar(nodes, group) {
    console.log('⭐ Star shape layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate star dimensions
    const starSize = Math.min(group._size[0], group._size[1]) * 0.35;
    const points = 5; // 5-pointed star
    const spacing = (2 * Math.PI) / nodes.length;
    
    console.log(`📏 Star layout:`, {
        centerX, centerY,
        starSize,
        points,
        spacing
    });
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        // Parametric star equation
        const angle = (i / nodes.length) * 2 * Math.PI;
        const radius = starSize * (1 + 0.3 * Math.sin(points * angle)); // Creates star shape
        
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        const oldPos = [...node.pos];
        node.pos[0] = x - width / 2;
        node.pos[1] = y - height / 2;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            angle: (angle * 180 / Math.PI).toFixed(1) + '°',
            radius: radius.toFixed(1)
        });
        
        applyBoundaryCheck(node, bounds, width, height);
    });
    
    console.log('✅ Star shape layout complete');
}

function arrangeSpiral(nodes, group) {
    console.log('🌀 Spiral layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate spiral dimensions
    const maxRadius = Math.min(group._size[0], group._size[1]) * 0.4;
    const rotations = 3; // Number of full rotations
    const spacing = (2 * Math.PI * rotations) / nodes.length;
    
    console.log(`📏 Spiral layout:`, {
        centerX, centerY,
        maxRadius,
        rotations,
        spacing
    });
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        // Archimedean spiral: r = a + bθ
        const angle = (i / nodes.length) * 2 * Math.PI * rotations;
        const radius = maxRadius * (angle / (2 * Math.PI * rotations)); // Increase radius with angle
        
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        const oldPos = [...node.pos];
        node.pos[0] = x - width / 2;
        node.pos[1] = y - height / 2;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            angle: (angle * 180 / Math.PI).toFixed(1) + '°',
            radius: radius.toFixed(1)
        });
        
        applyBoundaryCheck(node, bounds, width, height);
    });
    
    console.log('✅ Spiral layout complete');
}

function arrangeWave(nodes, group) {
    console.log('🌊 Wave layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate wave dimensions
    const waveWidth = group._size[0] * 0.7;
    const waveHeight = group._size[1] * 0.3;
    const waves = 2; // Number of wave cycles
    
    console.log(`📏 Wave layout:`, {
        centerX, centerY,
        waveWidth, waveHeight,
        waves
    });
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        // Sine wave pattern
        const x = centerX - waveWidth/2 + (i / (nodes.length - 1)) * waveWidth;
        const y = centerY + waveHeight * Math.sin((i / nodes.length) * 2 * Math.PI * waves);
        
        const oldPos = [...node.pos];
        node.pos[0] = x - width / 2;
        node.pos[1] = y - height / 2;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            position: (i / nodes.length).toFixed(2)
        });
        
        applyBoundaryCheck(node, bounds, width, height);
    });
    
    console.log('✅ Wave layout complete');
}

function arrangeFlower(nodes, group) {
    console.log('🌼 Flower layout starting...');
    
    const bounds = calculateLayoutBounds(group);
    const centerX = group._pos[0] + group._size[0] / 2;
    const centerY = group._pos[1] + group._size[1] / 2;
    
    // Calculate flower dimensions
    const flowerSize = Math.min(group._size[0], group._size[1]) * 0.35;
    const petals = 8; // Number of petals
    const spacing = (2 * Math.PI) / nodes.length;
    
    console.log(`📏 Flower layout:`, {
        centerX, centerY,
        flowerSize,
        petals,
        spacing
    });
    
    nodes.forEach((node, i) => {
        const width = node.size?.[0] || 200;
        const height = node.size?.[1] || 100;
        
        // Rose curve/polar flower equation
        const angle = (i / nodes.length) * 2 * Math.PI;
        const radius = flowerSize * Math.abs(Math.sin(petals * angle / 2));
        
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        const oldPos = [...node.pos];
        node.pos[0] = x - width / 2;
        node.pos[1] = y - height / 2;
        
        console.log(`   Node ${i}:`, {
            type: node.type || node.title,
            oldPos,
            newPos: node.pos,
            angle: (angle * 180 / Math.PI).toFixed(1) + '°',
            radius: radius.toFixed(1)
        });
        
        applyBoundaryCheck(node, bounds, width, height);
    });
    
    console.log('✅ Flower layout complete');
}

function arrangeGroup(group, layout) {
    console.log(`🎯 arrangeGroup called with layout: ${layout}`);
    console.log(`📦 Group info:`, {
        pos: group._pos,
        size: group._size,
        title: group.title || 'Untitled'
    });
    
    const nodes = getNodesInGroup(group);
    console.log(`🔍 Found ${nodes.length} nodes in group`);
    
    if (!nodes.length) {
        console.log('❌ No nodes to arrange');
        return;
    }

    nodes.forEach((node, i) => {
        console.log(`   Node ${i}:`, {
            type: node.type,
            pos: node.pos,
            size: node.size,
            title: node.title || node.type
        });
    });

    switch(layout) {
        case "grid":
            arrangeGrid(nodes, group);
            break;
        case "tight-grid":
            arrangeTightGrid(nodes, group);
            break;
        case "horizontal":
            arrangeHorizontal(nodes, group);
            break;
        case "vertical":
            arrangeVertical(nodes, group);
            break;
        case "circular":
            arrangeCircular(nodes, group);
            break;
        case "hierarchy-horizontal":
            arrangeHierarchyHorizontal(nodes, group);
            break;
        case "hierarchy-vertical":
            arrangeHierarchyVertical(nodes, group);
            break;
        case "triangle":
            arrangeTriangle(nodes, group);
            break;
        case "square":
            arrangeSquare(nodes, group);
            break;
        case "diamond":
            arrangeDiamond(nodes, group);
            break;
        case "fill-group":
            arrangeFillGroup(nodes, group);
            break;
        case "heart":
            arrangeHeart(nodes, group);
            break;
        case "star":
            arrangeStar(nodes, group);
            break;
        case "spiral":
            arrangeSpiral(nodes, group);
            break;
        case "wave":
            arrangeWave(nodes, group);
            break;
        case "flower":
            arrangeFlower(nodes, group);
            break;
        default:
            console.error(`❌ Unknown layout: ${layout}`);
            return;
    }

    console.log('🔄 Triggering canvas refresh');
    app.graph.setDirtyCanvas(true, true);
}

function arrangeUngroupedNodes(canvas, layout) {
    console.log(`🎯 arrangeUngroupedNodes called with layout: ${layout}`);
    
    // Handle the special "create-group" layout
    if (layout === "create-group") {
        createGroupForUngroupedNodes(canvas);
        return;
    }
    
    const ungroupedNodes = getUngroupedNodes();
    console.log(`🔍 Found ${ungroupedNodes.length} ungrouped nodes`);
    
    if (!ungroupedNodes.length) {
        console.log('❌ No ungrouped nodes to arrange');
        return;
    }
    
    // Get all groups to check for collisions
    const groups = app.graph._groups || [];
    
    // Create a virtual group that fits all ungrouped nodes with padding
    const virtualGroup = createVirtualGroupForUngrouped(canvas, ungroupedNodes);
    
    console.log(`📦 Virtual group created:`, {
        pos: virtualGroup._pos,
        size: virtualGroup._size
    });
    
    // Check if virtual group overlaps with any existing groups
    const overlappingGroups = groups.filter(group => {
        const vLeft = virtualGroup._pos[0];
        const vRight = vLeft + virtualGroup._size[0];
        const vTop = virtualGroup._pos[1];
        const vBottom = vTop + virtualGroup._size[1];
        
        const gLeft = group._pos[0];
        const gRight = gLeft + group._size[0];
        const gTop = group._pos[1];
        const gBottom = gTop + group._size[1];
        
        // Check for overlap
        return !(vRight < gLeft || vLeft > gRight || vBottom < gTop || vTop > gBottom);
    });
    
    // If virtual group overlaps with any groups, adjust its position
    if (overlappingGroups.length > 0) {
        console.log(`⚠️ Virtual group overlaps with ${overlappingGroups.length} existing groups, adjusting position`);
        
        // Find a safe area that doesn't overlap with any groups
        const canvasWidth = canvas.canvas.width || 3000;
        const canvasHeight = canvas.canvas.height || 2000;
        
        // Try to position in top-left corner first
        let safeX = 50;
        let safeY = 50;
        
        // Check if top-left is free
        const topLeftOverlaps = groups.some(group => {
            const gLeft = group._pos[0];
            const gRight = gLeft + group._size[0];
            const gTop = group._pos[1];
            const gBottom = gTop + group._size[1];
            
            const vRight = safeX + virtualGroup._size[0];
            const vBottom = safeY + virtualGroup._size[1];
            
            return !(vRight < gLeft || safeX > gRight || vBottom < gTop || safeY > gBottom);
        });
        
        if (topLeftOverlaps) {
            // Try bottom-right
            safeX = canvasWidth - virtualGroup._size[0] - 50;
            safeY = canvasHeight - virtualGroup._size[1] - 50;
            
            // Ensure it's within canvas bounds
            safeX = Math.max(50, safeX);
            safeY = Math.max(50, safeY);
        }
        
        virtualGroup._pos = [safeX, safeY];
        console.log(`   Adjusted virtual group position to:`, virtualGroup._pos);
    }
    
    ungroupedNodes.forEach((node, i) => {
        console.log(`   Ungrouped Node ${i}:`, {
            type: node.type,
            pos: node.pos,
            size: node.size,
            title: node.title || node.type
        });
    });
    
    switch(layout) {
        case "grid":
            arrangeGrid(ungroupedNodes, virtualGroup);
            break;
        case "tight-grid":
            arrangeTightGrid(ungroupedNodes, virtualGroup);
            break;
        case "horizontal":
            arrangeHorizontal(ungroupedNodes, virtualGroup);
            break;
        case "vertical":
            arrangeVertical(ungroupedNodes, virtualGroup);
            break;
        case "circular":
            arrangeCircular(ungroupedNodes, virtualGroup);
            break;
        case "hierarchy-horizontal":
            arrangeHierarchyHorizontal(ungroupedNodes, virtualGroup);
            break;
        case "hierarchy-vertical":
            arrangeHierarchyVertical(ungroupedNodes, virtualGroup);
            break;
        case "triangle":
            arrangeTriangle(ungroupedNodes, virtualGroup);
            break;
        case "square":
            arrangeSquare(ungroupedNodes, virtualGroup);
            break;
        case "diamond":
            arrangeDiamond(ungroupedNodes, virtualGroup);
            break;
        case "fill-group":
            arrangeFillGroup(ungroupedNodes, virtualGroup);
            break;
        case "heart":
            arrangeHeart(ungroupedNodes, virtualGroup);
            break;
        case "star":
            arrangeStar(ungroupedNodes, virtualGroup);
            break;
        case "spiral":
            arrangeSpiral(ungroupedNodes, virtualGroup);
            break;
        case "wave":
            arrangeWave(ungroupedNodes, virtualGroup);
            break;
        case "flower":
            arrangeFlower(ungroupedNodes, virtualGroup);
            break;
        default:
            console.error(`❌ Unknown layout: ${layout}`);
            return;
    }
    
    // After arranging, check if any nodes ended up inside groups and move them if needed
    const nodesInsideGroupsAfter = ungroupedNodes.filter(node => {
        if (!node.pos) return false;
        
        const w = node.size?.[0] || 200;
        const h = node.size?.[1] || 100;
        const cx = node.pos[0] + w / 2;
        const cy = node.pos[1] + h / 2;
        
        return groups.some(group => {
            return (
                cx >= group._pos[0] &&
                cx <= group._pos[0] + group._size[0] &&
                cy >= group._pos[1] &&
                cy <= group._pos[1] + group._size[1]
            );
        });
    });
    
    if (nodesInsideGroupsAfter.length > 0) {
        console.log(`⚠️ ${nodesInsideGroupsAfter.length} nodes ended up inside groups, moving them out`);
        
        nodesInsideGroupsAfter.forEach(node => {
            // Move node to the right of the virtual group
            const w = node.size?.[0] || 200;
            const newX = virtualGroup._pos[0] + virtualGroup._size[0] + 20;
            const newY = virtualGroup._pos[1] + 50;
            
            console.log(`   Moving node ${node.type || node.title} to: [${newX}, ${newY}]`);
            node.pos[0] = newX;
            node.pos[1] = newY;
        });
    }

    console.log('🔄 Triggering canvas refresh');
    app.graph.setDirtyCanvas(true, true);
}

// Main extension registration with both features integrated
app.registerExtension({
    name: "Comfy.GroupArrangeMenu",

    setup() {
        console.log("📐 Group arrange (canvas-driven) menu loaded");

        const original = LGraphCanvas.prototype.getCanvasMenuOptions;

        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = original ? original.call(this) : [];
            const group = getGroupUnderMouse(this);
            
            // Enhanced selected nodes detection
            let selectedNodes = [];
            let hasSelectedNodes = false;
            
            // Try multiple methods to detect selected nodes
            if (this.selected_nodes && this.selected_nodes.length > 0) {
                selectedNodes = this.selected_nodes;
                hasSelectedNodes = true;
                console.log(`📊 Menu: Found ${selectedNodes.length} selected nodes in canvas.selected_nodes`);
            } else if (app.graph && app.graph._nodes) {
                // Check node flags
                selectedNodes = app.graph._nodes.filter(node => node.flags?.selected === true);
                hasSelectedNodes = selectedNodes.length > 0;
                console.log(`📊 Menu: Found ${selectedNodes.length} selected nodes via node.flags`);
            }
            
            console.log("📊 Menu context check:", {
                group: group ? "Yes" : "No",
                hasSelectedNodes,
                selectedCount: selectedNodes.length,
                canvasType: this.constructor.name
            });
            
            // Determine import option text based on context
            const importOptionText = group 
                ? `📥 Import Workflow into "${group.title}"` 
                : "📥 Import Workflow into Group";
            
            // Add Import Workflow option (always visible)
            options.push(null);
            options.push({
                content: importOptionText,
                callback: (_, __, event) => {
                    const pos = this.convertEventToCanvasOffset(event);
                    // If we're right-clicking over a group, import into that group
                    const targetGroup = group || null;
                    importWorkflowIntoGroup(pos, targetGroup);
                }
            });
            
            // Only show "Arrange Ungrouped Nodes" when NOT inside a group
            if (!group) {
                options.push(null);
                options.push({
                    content: "📐 Arrange Ungrouped Nodes",
                    submenu: {
                        options: [
                            { content: "⊞ Grid", callback: () => arrangeUngroupedNodes(this, "grid") },
                            { content: "⩩ Tight Grid (max 20px spacing)", callback: () => arrangeUngroupedNodes(this, "tight-grid") },
                            { content: "─ Horizontal", callback: () => arrangeUngroupedNodes(this, "horizontal") },
                            { content: "┃ Vertical", callback: () => arrangeUngroupedNodes(this, "vertical") },
                            { content: "⭕️ Circular", callback: () => arrangeUngroupedNodes(this, "circular") },
                            { content: "△ Triangle", callback: () => arrangeUngroupedNodes(this, "triangle") },
                            { content: "☐ Square", callback: () => arrangeUngroupedNodes(this, "square") },
                            { content: "💎 Diamond", callback: () => arrangeUngroupedNodes(this, "diamond") },
                            null,
                            { content: "❤️ Heart", callback: () => arrangeUngroupedNodes(this, "heart") },
                            { content: "⭐ Star", callback: () => arrangeUngroupedNodes(this, "star") },
                            { content: "🌀 Spiral", callback: () => arrangeUngroupedNodes(this, "spiral") },
                            { content: "🌊 Wave", callback: () => arrangeUngroupedNodes(this, "wave") },
                            { content: "🌼 Flower", callback: () => arrangeUngroupedNodes(this, "flower") },
                            null,
                            { content: "Fill Group (L→R, T→B)", callback: () => arrangeUngroupedNodes(this, "fill-group") },
                            null,
                            { content: "↔ Hierarchy (Horizontal)", callback: () => arrangeUngroupedNodes(this, "hierarchy-horizontal") },
                            { content: "↕ Hierarchy (Vertical)", callback: () => arrangeUngroupedNodes(this, "hierarchy-vertical") }
                        ]
                    }
                });
                
                // Add grouping options
                options.push({
                    content: "📦 Group All Ungrouped Nodes",
                    callback: () => createGroupForUngroupedNodes(this)
                });
                
                // Add option to group selected nodes (if any are selected)
                if (hasSelectedNodes) {
                    console.log(`📊 Adding menu option for ${selectedNodes.length} selected nodes`);
                    options.push({
                        content: `📦 Group Selected Nodes (${selectedNodes.length})`,
                        callback: () => createGroupForSelectedNodes(this)
                    });
                } else {
                    console.log('📊 No selected nodes found, not adding "Group Selected Nodes" option');
                }
            }
            
            // Only show "Arrange This Group" when inside a group
            if (group) {
                options.push(null);
                options.push({
                    content: "📐 Arrange This Group",
                    submenu: {
                        options: [
                            { content: "⊞ Grid", callback: () => arrangeGroup(group, "grid") },
                            { content: "⩩ Tight Grid (max 20px spacing)", callback: () => arrangeGroup(group, "tight-grid") },
                            { content: "─ Horizontal", callback: () => arrangeGroup(group, "horizontal") },
                            { content: "┃ Vertical", callback: () => arrangeGroup(group, "vertical") },
                            { content: "⭕️ Circular", callback: () => arrangeGroup(group, "circular") },
                            { content: "△ Triangle", callback: () => arrangeGroup(group, "triangle") },
                            { content: "☐ Square", callback: () => arrangeGroup(group, "square") },
                            { content: "💎 Diamond", callback: () => arrangeGroup(group, "diamond") },
                            null,
                            { content: "❤️ Heart", callback: () => arrangeGroup(group, "heart") },
                            { content: "⭐ Star", callback: () => arrangeGroup(group, "star") },
                            { content: "🌀 Spiral", callback: () => arrangeGroup(group, "spiral") },
                            { content: "🌊 Wave", callback: () => arrangeGroup(group, "wave") },
                            { content: "🌼 Flower", callback: () => arrangeGroup(group, "flower") },
                            null,
                            { content: "Fill Group (L→R, T→B)", callback: () => arrangeGroup(group, "fill-group") },
                            null,
                            { content: "↔ Hierarchy (Horizontal)", callback: () => arrangeGroup(group, "hierarchy-horizontal") },
                            { content: "↕ Hierarchy (Vertical)", callback: () => arrangeGroup(group, "hierarchy-vertical") }
                        ]
                    }
                });
            }

            return options;
        };
    }
});