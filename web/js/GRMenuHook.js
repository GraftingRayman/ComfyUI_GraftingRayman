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
        
        if (typeof LiteGraph.LGraphGroup === 'undefined') {
            console.error('❌ LiteGraph.LGraphGroup is not defined');
            // Try alternative names
            if (typeof LGraphGroup !== 'undefined') {
                console.log('⚠️ Found LGraphGroup (without LiteGraph prefix)');
                var GroupClass = LGraphGroup;
            } else {
                console.error('❌ No group class found');
                return;
            }
        } else {
            var GroupClass = LiteGraph.LGraphGroup;
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

app.registerExtension({
    name: "Comfy.GroupArrangeMenu",

    setup() {
        console.log("📐 Group arrange (canvas-driven) menu loaded");

        const original = LGraphCanvas.prototype.getCanvasMenuOptions;

        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = original ? original.call(this) : [];
            const group = getGroupUnderMouse(this);
            
            // Only show "Arrange Ungrouped Nodes" when NOT inside a group
            if (!group) {
                options.push(null);
                options.push({
                    content: "📐 Arrange Ungrouped Nodes",
                    submenu: {
                        options: [
                            { content: "Grid", callback: () => arrangeUngroupedNodes(this, "grid") },
                            { content: "Tight Grid (max 20px spacing)", callback: () => arrangeUngroupedNodes(this, "tight-grid") },
                            { content: "Horizontal", callback: () => arrangeUngroupedNodes(this, "horizontal") },
                            { content: "Vertical", callback: () => arrangeUngroupedNodes(this, "vertical") },
                            { content: "Circular", callback: () => arrangeUngroupedNodes(this, "circular") },
                            null,
                            { content: "Hierarchy (Horizontal)", callback: () => arrangeUngroupedNodes(this, "hierarchy-horizontal") },
                            { content: "Hierarchy (Vertical)", callback: () => arrangeUngroupedNodes(this, "hierarchy-vertical") }
                        ]
                    }
                });
                
                // Add new option to create group for ungrouped nodes
                options.push({
                    content: "📦 Group All Ungrouped Nodes",
                    callback: () => createGroupForUngroupedNodes(this)
                });
            }
            
            // Only show "Arrange This Group" when inside a group
            if (group) {
                options.push(null);
                options.push({
                    content: "📐 Arrange This Group",
                    submenu: {
                        options: [
                            { content: "Grid", callback: () => arrangeGroup(group, "grid") },
                            { content: "Tight Grid (max 20px spacing)", callback: () => arrangeGroup(group, "tight-grid") },
                            { content: "Horizontal", callback: () => arrangeGroup(group, "horizontal") },
                            { content: "Vertical", callback: () => arrangeGroup(group, "vertical") },
                            { content: "Circular", callback: () => arrangeGroup(group, "circular") },
                            null,
                            { content: "Hierarchy (Horizontal)", callback: () => arrangeGroup(group, "hierarchy-horizontal") },
                            { content: "Hierarchy (Vertical)", callback: () => arrangeGroup(group, "hierarchy-vertical") }
                        ]
                    }
                });
            }

            return options;
        };
    }
});